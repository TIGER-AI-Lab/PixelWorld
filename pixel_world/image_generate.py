import os
import random
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from tqdm import tqdm
import pandas as pd
import dataframe_image as dfi
import json

def render_table_image(table_dict, output_path="table_image.png"):
    """
    使用 dataframe_image 将 Pandas DataFrame 导出为图片，
    支持隐藏数字列名并加粗第一行（作为原始列名）。
    """
    import pandas as pd
    import dataframe_image as dfi

    # 1) 用数字 0 到 n 替代列名
    column_count = len(table_dict["columns"])
    numeric_columns = list(range(column_count))  # 创建数字列名
    data_with_columns = [table_dict["columns"]] + table_dict["data"]  # 添加原列名作为第一行

    # 2) 构造 DataFrame
    df = pd.DataFrame(data_with_columns, columns=numeric_columns)

    # 3) 使用 Styler 来定制表格外观
    def highlight_first_row(row):
        """对第一行应用加粗样式"""
        if row.name == 0:  # 第一行
            return ['font-weight: bold' for _ in row]
        return ['' for _ in row]

    df_styled = (
        df.style
          .apply(highlight_first_row, axis=1)  # 对第一行加粗
          .set_properties(**{
              'font-size': '12px',   # 设置字体大小
              'padding': '2px 6px',  # 设置单元格内边距
          })
          .set_table_styles([
              {
                  'selector': 'th, td',
                  'props': [
                      ('border', '1px solid black'),     # 设置边框样式
                      ('border-collapse', 'collapse'),   # 去掉表格缝隙
                  ]
              },
              {
                  'selector': 'th',
                  'props': [
                      ('background-color', '#ddd'),      # 设置表头背景色
                      ('font-weight', 'bold')
                  ]
              }
          ])
          .hide(axis='columns')  # 隐藏数字列名
    )
    dfi.export(df_styled, output_path, max_rows=-1, max_cols=-1, table_conversion='selenium')

class ImageGenerator(object):
    def __init__(self, 
                 random_image_scale=True, 
                 random_font_size=True, 
                 random_padding=True, 
                 random_font=True, 
                 add_low_freq_noise=True, 
                 add_high_freq_noise=True,
                 available_fonts=None,
                 no_random=False,):
        """
        初始化ImageGenerator并设置各类随机化开关。
        """
        self.random_image_scale = random_image_scale
        self.random_font_size = random_font_size
        self.random_padding = random_padding
        self.random_font = random_font
        self.add_low_freq_noise = add_low_freq_noise
        self.add_high_freq_noise = add_high_freq_noise

        # 默认范围
        self.image_width_range = (512, 1024)
        # self.image_height_range = (512, 2048)
        self.image_height = 256
        # TODO: Set the size of image be limited, and
        self.font_size_range = (15, 25)
        self.padding_range = (5, 30)

        # 默认字体
        if available_fonts is None or len(available_fonts) == 0:
            self.available_fonts = ["/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
        else:
            self.available_fonts = available_fonts
        self.dataset_cache = {}
        self.no_random = no_random
        self.default_params = {
            "image_width": 512,
            "image_height": 256,
            "font_size": 15,
            "padding": 20,
            "font_path": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "line_spacing": 10,
        }

    def generate_image(self, text, output_image_path, embeded_img_path=None, force=False):
        if os.path.exists(output_image_path) and not force:
            return  # 若文件已存在则跳过

        # 如果 no_random 为 True，直接使用默认参数
        if self.no_random:
            params = self.default_params
            image_width = params["image_width"]
            image_height = params["image_height"]
            font_size = params["font_size"]
            padding = params["padding"]
            font_path = params["font_path"]
            # line_spacing = params["line_spacing"]
        else:
            # 随机图像尺寸
            image_width = random.randint(*self.image_width_range) if self.random_image_scale else 512
            image_height = self.image_height

            # 随机字体选择
            font_path = random.choice(self.available_fonts) if self.random_font and self.available_fonts else \
            self.available_fonts[0]

            # 随机字体大小
            font_size = random.randint(*self.font_size_range) if self.random_font_size else 30

            # 随机 Padding
            padding = random.randint(*self.padding_range) if self.random_padding else 50

        # Line spacing
        base_line_spacing = 10
        try:
            font = ImageFont.truetype(font_path, font_size)
            _, _, _, line_height = font.getbbox("A")
        except Exception:
            font = ImageFont.load_default()
            _, _, _, line_height = font.getbbox("A")
        line_spacing = line_height + base_line_spacing

        # 加载字体
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 先将文本按照 "<img>" 分割
        text_split = text.split("<img>")
        has_embed = (embeded_img_path is not None) and ("<img>" in text)

        # 计算可用宽度
        max_width = image_width - 2 * padding

        if has_embed:
            # 取出上、下两部分文本
            before_text = text_split[0]
            after_text = text_split[1] if len(text_split) > 1 else ""

            # 打开内嵌图片并检查宽度，如果超过可用宽度则等比例缩放
            embed_img = Image.open(embeded_img_path)
            embed_w, embed_h = embed_img.size

            if image_width < embed_w:
                image_width = int(
                    embed_w * random.uniform(1, 1.2)) + 2 * padding if not self.no_random else embed_w + 2 * padding
                max_width = image_width - 2 * padding

            # 分别对上下文本进行换行处理
            lines_before = self._wrap_text(before_text, font, max_width)
            lines_after = self._wrap_text(after_text, font, max_width)

            if embed_w > max_width:
                ratio = max_width / float(embed_w)
                new_width = max_width
                new_height = int(embed_h * ratio)
                embed_img = embed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                embed_w, embed_h = new_width, new_height

            total_height = line_spacing * len(lines_before) + embed_h + line_spacing * len(lines_after) + 2 * padding
        else:
            # 如果没有内嵌图片，直接正常处理文本
            lines_before = self._wrap_text(text, font, max_width)
            lines_after = []
            embed_img = None
            embed_w = 0
            embed_h = 0

            total_height = line_spacing * len(lines_before) + 2 * padding

        if total_height > image_height:
            image_height = total_height

        # 创建白色背景图像
        image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))

        # 添加低频和高频噪声（仅在 no_random 为 False 时添加）
        if not self.no_random:
            if self.add_low_freq_noise:
                image = self._add_low_freq_noise(image)
            if self.add_high_freq_noise:
                image = self._add_high_freq_noise(image)
        # if self.add_low_freq_noise:
        #     image = self._add_low_freq_noise(image)
        # if self.add_high_freq_noise:
        #     image = self._add_high_freq_noise(image)

        # 绘制文本和内嵌图片
        draw = ImageDraw.Draw(image)
        y_position = padding

        # 先绘制上部分文本
        for line in lines_before:
            draw.text((padding, y_position), line, fill=(0, 0, 0), font=font)
            y_position += line_spacing

        # 如果有内嵌图片，则插入
        if has_embed and embed_img:
            x_position = padding if embed_w >= max_width else (image_width - embed_w) // 2
            image.paste(embed_img, (x_position, y_position))
            y_position += embed_h

        # 再绘制下部分文本
        for line in lines_after:
            draw.text((padding, y_position), line, fill=(0, 0, 0), font=font)
            y_position += line_spacing

        # 保存图像
        image.save(output_image_path)

    def construct_image_dataset(self, dataset_name, dataset_list, query_text_template, cot_flag, mode="base"):
        self.dataset_cache[dataset_name] = dataset_list
        data_dict = {}
        print(f"Constructing image dataset: {dataset_name}")
        last_layer_folder = "cot" if cot_flag else "base"
        os.makedirs(f"./image_cache/{dataset_name}/{last_layer_folder}", exist_ok=True)
        for task, records in dataset_list.items():
            data_with_vision = []
            task_dir = f"./image_cache/{dataset_name}/{last_layer_folder}/{task}"
            os.makedirs(task_dir, exist_ok=True)

            for idx, (question, answer) in tqdm(enumerate(records)):
                if mode == "base":
                    output_image_path = os.path.join(task_dir, f"{idx}.png")
                    text_for_image = question
                    self.generate_image(text_for_image, output_image_path)

                    query_text = query_text_template
                    data_with_vision.append([query_text, output_image_path, answer])
                else:
                    query_text, text_for_table = question.split("######")
                    table_path = os.path.join(task_dir, f"{idx}_table.png")
                    # Not exist or can't open
                    # print(question)
                    # print("#" * 20)
                    if not os.path.exists(table_path) or not Image.open(table_path):
                        # print(query_text)
                        # print("#" * 20)
                        # print(text_for_table)
                        # print("#" * 20)
                        table_dict = json.loads(text_for_table)
                        render_table_image(table_dict, table_path)
                    if mode == "table_semi":
                        output_image_path = os.path.join(task_dir, f"{idx}_semi.png")
                        self.generate_image("<img>", output_image_path, embeded_img_path=table_path)
                        data_with_vision.append([query_text, output_image_path, answer])
                        # print([query_text, output_image_path, answer])
                        # exit(0)
                    elif mode == "table_img":
                        output_image_path = os.path.join(task_dir, f"{idx}_full.png")
                        self.generate_image(query_text, output_image_path, embeded_img_path=table_path)
                        data_with_vision.append([query_text_template, output_image_path, answer])
                        # print([query_text_template, output_image_path, answer])
                        # exit(0)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

            data_dict[task] = data_with_vision
        print("Finish construction...")
        return data_dict

    # If image is broken, regenerate it
    def regenerate_image(self, output_image_path):
        # Parse dataset_name, task_name, image_id from image_path
        dataset_name, task_name, image_id = output_image_path.split("/")[-3:]
        image_id = image_id.split(".")[0]
        question, answer = self.dataset_cache[dataset_name][task_name][int(image_id)]
        text_for_image = question
        self.generate_image(text_for_image, output_image_path, force=True)

    def subsample_image(self, image_path, max_pixel, model_name):
        if max_pixel <= 0:
            raise ValueError("max_pixel must be greater than 0")
        base, ext = os.path.splitext(image_path)
        new_path = f"{base}_{model_name}{ext}"
        try:
            image = Image.open(image_path)
        except Exception as e:
            self.regenerate_image(image_path)
            try:
                image = Image.open(image_path)
            except Exception as e:
                # Raise Exception if still cannot open the image
                raise e
        # image = Image.open(image_path)
        width, height = image.size
        if width * height > max_pixel:
            scale = (max_pixel / (width * height)) ** 0.5
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = image.resize((new_width, new_height), Image.LANCZOS)
            image.save(new_path)
            return new_path
        return image_path

    def _wrap_text(self, text, font, max_width):
        """
        将文本根据像素宽度进行折行。
        先用textwrap进行粗略分行，然后逐词检查宽度。
        """
        lines = []
        for paragraph in text.splitlines():
            if paragraph.strip():
                # 使用textwrap进行粗略分行
                wrapped_lines = textwrap.wrap(paragraph, width=1000)
                for line in wrapped_lines:
                    split_line = ""
                    for word in line.split():
                        temp_line = (split_line + " " + word).strip()
                        w = font.getbbox(temp_line)[2]
                        if w <= max_width:
                            split_line = temp_line
                        else:
                            if split_line:
                                lines.append(split_line)
                            split_line = word
                    if split_line:
                        lines.append(split_line)
            else:
                # 空行
                lines.append("")
            # 段落后加一空行
            lines.append("")
        # print(lines)
        return lines

    def _generate_radial_noise(self, width, height):
        """
        基于随机中心点生成径向噪声层（L模式），距离中心越远亮度越低。
        """
        noise_layer = Image.new("L", (width, height), 0)
        pixels = noise_layer.load()
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)

        max_dist = ((width ** 2 + height ** 2) ** 0.5) / 2
        for y in range(height):
            for x in range(width):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                val = 255 - int(min(dist, 255))
                pixels[x, y] = val
        return noise_layer

    def _generate_horizontal_noise(self, width, height):
        """
        横向渐变噪声层，从左到右或从右到左渐变。
        """
        noise_layer = Image.new("L", (width, height), 0)
        pixels = noise_layer.load()
        direction = random.choice(["left-to-right", "right-to-left"])
        for x in range(width):
            val = int((x / (width - 1)) * 255)
            if direction == "right-to-left":
                val = 255 - val
            for y in range(height):
                pixels[x, y] = val
        return noise_layer

    def _generate_vertical_noise(self, width, height):
        """
        纵向渐变噪声层，从上到下或从下到上渐变。
        """
        noise_layer = Image.new("L", (width, height), 0)
        pixels = noise_layer.load()
        direction = random.choice(["top-to-bottom", "bottom-to-top"])
        for y in range(height):
            val = int((y / (height - 1)) * 255)
            if direction == "bottom-to-top":
                val = 255 - val
            for x in range(width):
                pixels[x, y] = val
        return noise_layer

    def _generate_multiple_gaussians_noise(self, width, height):
        """
        多高斯点噪声层：随机在图中放置多个高斯亮点。
        """
        noise_layer = Image.new("L", (width, height), 0)
        pixels = noise_layer.load()
        num_points = random.randint(3, 6)
        for _ in range(num_points):
            gx = random.randint(0, width - 1)
            gy = random.randint(0, height - 1)
            radius = random.randint(20, 50)
            peak_val = 255
            for y in range(max(0, gy - radius), min(height, gy + radius)):
                for x in range(max(0, gx - radius), min(width, gx + radius)):
                    dist = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
                    if dist < radius:
                        val = int(peak_val * (1 - dist / radius))
                        pixels[x, y] = max(pixels[x, y], val)
        return noise_layer

    def _add_low_freq_noise(self, image, modes=None):
        """
        添加低频噪声。
        modes为一个列表，包含以下任意模式，可多选：
        ["radial", "horizontal", "vertical", "multiple-gaussians"]

        若不指定modes，则随机选择1到4个模式进行混合。
        最终混合后，以0.3叠加到原图上。
        """
        image = image.convert("RGB")
        width, height = image.size

        available_modes = ["radial", "horizontal", "vertical", "multiple-gaussians"]
        if modes is None or len(modes) == 0:
            # 随机选择n个模式
            n = random.randint(1, 4)
            modes = random.sample(available_modes, n)

        # 生成每种模式的噪声层并进行叠加
        noise_arrays = []
        for m in modes:
            if m == "radial":
                layer = self._generate_radial_noise(width, height)
            elif m == "horizontal":
                layer = self._generate_horizontal_noise(width, height)
            elif m == "vertical":
                layer = self._generate_vertical_noise(width, height)
            elif m == "multiple-gaussians":
                layer = self._generate_multiple_gaussians_noise(width, height)
            else:
                continue

            # 对每个单独噪声层进行高斯模糊，让其低频化
            layer = layer.filter(ImageFilter.GaussianBlur(radius=30))
            noise_arrays.append(np.array(layer, dtype=np.float32))

        # 将多个噪声层叠加，这里采用求和再归一化的方式
        if len(noise_arrays) == 0:
            return image  # 没有噪声层则直接返回原图

        combined = np.sum(noise_arrays, axis=0)
        # 归一化到[0, 255]
        combined = combined / combined.max() * 255
        combined = combined.astype(np.uint8)
        noise_layer = Image.fromarray(combined, mode="L")

        # 将单通道扩展为RGB
        noise_colored = Image.merge("RGB", (noise_layer, noise_layer, noise_layer))

        # 将噪声以alpha=0.3叠加到原图上
        result = Image.blend(image, noise_colored, alpha=0.3)
        return result

    def _add_high_freq_noise(self, image, sigma=20):
        """
        添加高频噪声，使用高斯分布的噪声而不是均匀分布。
        高斯噪声可正可负，从而既可能使局部变亮也可能变暗。

        参数：
        - sigma: 高斯噪声的标准差，可根据需要调整。

        将噪声添加到图像中并clip到[0,255]。
        """
        image = image.convert("RGB")
        arr = np.array(image).astype(np.float32)

        # 生成高斯噪声（均值0，标准差sigma）
        noise = np.random.normal(loc=0, scale=sigma, size=arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    # This function is not used in the current implementation
    def _add_frequency_domain_noise(self, image, noise_level=100, mode='all'):
        """
        对图像进行FFT后在频域添加噪声，再通过IFFT返回空间域。

        参数:
        - image: PIL Image对象
        - noise_level: 噪声强度，可根据需要调整
        - mode: 噪声添加模式
            'all'：对所有频率统一添加噪声
            'low'：仅对低频分量添加较强噪声
            'high'：仅对高频分量添加较强噪声
        """
        image = image.convert("RGB")
        arr = np.array(image, dtype=np.float32)

        # 分离通道进行FFT
        # FFT的零频分量在左上角，需要搬移到中心用fftshift处理
        # 不过在添加噪声时，无论是否shift，只要一致对待即可
        channels_fft = []
        for c in range(3):
            f = np.fft.fft2(arr[:, :, c])  # FFT
            fshift = np.fft.fftshift(f)  # 将频谱中心化，低频在中心，高频在四周
            channels_fft.append(fshift)

        height, width = arr.shape[:2]

        # 准备频率坐标系，用于构造频率掩模
        cy, cx = height // 2, width // 2
        y_indices, x_indices = np.indices((height, width))
        # 距离中心频率点的半径
        radius = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        max_radius = np.sqrt((cx) ** 2 + (cy) ** 2)

        # 根据mode来控制噪声分布
        # 这里以gaussian噪声为例（均值0），由noise_level控制噪声幅度
        # 您可以自定义噪声分布和mask
        if mode == 'all':
            # 所有频率统一添加噪声
            mask = np.ones((height, width), dtype=np.float32)
        elif mode == 'low':
            # 对低频区添加更强噪声，离中心越近噪声越强
            # 构造一个反比例关系，比如(1 - radius/max_radius)作为权重
            mask = 1 - (radius / max_radius)
        elif mode == 'high':
            # 对高频区添加更强噪声，离中心越远噪声越强
            mask = (radius / max_radius)
        else:
            # 未知模式则全频添加
            mask = np.ones((height, width), dtype=np.float32)

        # 为了给实部和虚部添加噪声，可以生成复数噪声：
        # 生成两个高斯分布的实部和虚部噪声，然后组合为复数。
        # 噪声幅度根据mask和noise_level调节。
        real_noise = np.random.normal(loc=0, scale=noise_level, size=(height, width))
        imag_noise = np.random.normal(loc=0, scale=noise_level, size=(height, width))
        complex_noise = (real_noise + 1j * imag_noise) * mask

        # 对每个通道添加噪声
        noisy_channels = []
        for fshift in channels_fft:
            fshift_noisy = fshift + complex_noise
            # 逆shift和逆FFT
            f_ishift = np.fft.ifftshift(fshift_noisy)
            img_back = np.fft.ifft2(f_ishift)
            # 取实部作为还原后的图像通道
            img_back = np.real(img_back)
            noisy_channels.append(img_back)

        # 合并通道并裁剪为[0,255]
        noisy_arr = np.stack(noisy_channels, axis=-1)
        noisy_arr = np.clip(noisy_arr, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_arr, mode="RGB")

class ImageGenerator_withTextHeatmap(ImageGenerator):
    def __init__(self,
                 random_image_scale=True,
                 random_font_size=True,
                 random_padding=True,
                 random_font=True,
                 add_low_freq_noise=True,
                 add_high_freq_noise=True,
                 available_fonts=None,
                 no_random=False):
        super().__init__(random_image_scale, random_font_size, random_padding, random_font, add_low_freq_noise,
                         add_high_freq_noise, available_fonts, no_random)

    def process_text_fromDecoder(self, text):
        text = text.replace("Ġ", " ")
        text = text.replace("Ċ", "\n")
        text = text.replace("ċ", "\t")
        text = text.replace("ĉ", "\r")
        text = text.replace("Ĉ", "\r\n")
        return text

    def generate_image(self, text, output_image_path, embeded_img_path=None, force=False):
        # 1. Replace "Ġ" with " " for display
        text = self.process_text_fromDecoder(text)
        super().generate_image(text, output_image_path, embeded_img_path, force)

    def generate_image_withTextHeatmap(self, text, output_image_path, text_tokens, text_heatmap, force=False):
        """
        Generate an image with token-level heatmap visualization.
        - text: the raw text (we assume no <img> placeholder)
        - text_tokens: list of tokens, e.g. ["Hello", ",", "this", ...]
        - text_heatmap: list of float values (same length as text_tokens)
        """
        if os.path.exists(output_image_path) and not force:
            return  # Skip if file exists

        # Preprocess text and tokens
        text = self.process_text_fromDecoder(text)
        text_tokens = [self.process_text_fromDecoder(tok) for tok in text_tokens]

        # Use default parameters if no_random is True
        if self.no_random:
            params = self.default_params
            image_width = params["image_width"]
            image_height = params["image_height"]
            font_size = params["font_size"]
            padding = params["padding"]
            font_path = params["font_path"]
            # line_spacing = params["line_spacing"]
        else:
            # Random image size
            image_width = random.randint(*self.image_width_range) if self.random_image_scale else 512
            image_height = self.image_height

            # Random font choice
            font_path = random.choice(self.available_fonts) if self.random_font and self.available_fonts else \
            self.available_fonts[0]

            # Random font size
            font_size = random.randint(*self.font_size_range) if self.random_font_size else 30

            # Random padding
            padding = random.randint(*self.padding_range) if self.random_padding else 50

        # Line spacing
        base_line_spacing = 10
        try:
            font = ImageFont.truetype(font_path, font_size)
            _, _, _, line_height = font.getbbox("A")
        except Exception:
            font = ImageFont.load_default()
            _, _, _, line_height = font.getbbox("A")
        line_spacing = line_height + base_line_spacing

        # Load font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # Compute max width for text layout
        max_width = image_width - 2 * padding

        # Layout each token with a heatmap
        wrapped_token_lines = self._wrap_tokens_with_heatmap_tokens(text_tokens, text_heatmap, font, max_width)

        # Compute total height
        num_lines = len(wrapped_token_lines)
        total_height = num_lines * line_spacing + 2 * padding
        if total_height > image_height:
            image_height = total_height

        # Create the image canvas
        image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))

        # Add noise if no_random is False
        if not self.no_random:
            if self.add_low_freq_noise:
                image = self._add_low_freq_noise(image)
            if self.add_high_freq_noise:
                image = self._add_high_freq_noise(image)

        draw = ImageDraw.Draw(image)

        # Draw each token with color-coded background
        y_position = padding
        for line_info in wrapped_token_lines:
            x_position = padding
            for (tok_str, heat_val, tok_w) in line_info:
                # Normalize heatmap values to [0, 1]
                color_factor = max(min(heat_val, 1.0), 0.0)
                # White to red gradient for background
                R = 255
                G = int(255 * (1 - color_factor))
                B = int(255 * (1 - color_factor))

                # Draw background rectangle
                draw.rectangle([x_position, y_position, x_position + tok_w, y_position + line_height],
                               fill=(R, G, B))

                # Draw token text in black
                draw.text((x_position, y_position), tok_str, fill=(0, 0, 0), font=font)

                x_position += tok_w

            y_position += line_spacing

        # Save the image
        image.save(output_image_path)

    def text_heatmap(self, text_tokens, text_heatmap):
        """
        Print each token & its heatmap value in console.
        """
        for tok, val in zip(text_tokens, text_heatmap):
            display_tok = tok.replace("Ġ", " ")
            print(f"Token: '{display_tok}' => Heat: {val:.4f}")

    def _wrap_tokens_with_heatmap_tokens(self, tokens, heatmap_values, font, max_width):
        """
        Process tokens and their heatmap values to create lines that fit within the max width.
        Handles tokens based on word boundaries and newlines.
        Each line is a list of tuples: (character, heatmap_value, character_width).
        """
        lines = []
        current_line = []
        current_width = 0

        for tok, val in zip(tokens, heatmap_values):
            # Split the token by spaces to identify potential breakpoints
            token_parts = tok.split(" ")

            for i, part in enumerate(token_parts):
                if i > 0:  # Add a space before the part if it's not the first part of the token
                    part = " " + part

                # Measure the width of the entire part
                bbox = font.getbbox(part)
                part_width = bbox[2] - bbox[0]

                # Handle forced line breaks within the token
                if "\n" in part:
                    sub_parts = part.split("\n")
                    for j, sub_part in enumerate(sub_parts):
                        if sub_part:
                            # Measure width of sub-part
                            sub_bbox = font.getbbox(sub_part)
                            sub_part_width = sub_bbox[2] - sub_bbox[0]

                            # Check if the sub-part fits in the current line
                            if current_width + sub_part_width > max_width:
                                # Add the current line to the result and start a new line
                                lines.append(current_line)
                                current_line = []
                                current_width = 0

                            # Add the sub-part to the current line
                            current_line.append((sub_part, val, sub_part_width))
                            current_width += sub_part_width

                        # Handle forced line break (add the current line to the result)
                        lines.append(current_line)
                        current_line = []
                        current_width = 0

                else:
                    # Check if the part fits in the current line
                    if current_width + part_width > max_width:
                        # Allow breaking at spaces only (if not the first part)
                        if current_line:
                            lines.append(current_line)
                            current_line = []
                            current_width = 0

                    # Add the part to the current line
                    current_line.append((part, val, part_width))
                    current_width += part_width

        # Add the last line if it contains any characters
        if current_line:
            lines.append(current_line)

        return lines


if __name__ == "__main__":
    # Initialize the image generator with various randomization options
    generator = ImageGenerator_withTextHeatmap(
        random_image_scale=True,
        random_font_size=True,
        random_padding=True,
        random_font=True,
        add_low_freq_noise=True,
        add_high_freq_noise=True,
        no_random=True,
    )

    # Define text tokens and their heatmap values
    text_tokens = [
        'The', 'Ġcorrect', 'Ġanswer', 'Ġis', 'ĠB', ')', 'Ġ', '7', '.ĊĊ',
        'Reason', 'ing', ':', 'ĠThe', 'Ġoperation', 'Ġin', 'Ġthe', 'Ġquestion',
        'Ġis', 'Ġaddition', ',', 'Ġwhich', 'Ġmeans', 'Ġinvolves', 'Ġcombining',
        'Ġtwo', 'Ġor', 'Ġmore', 'Ġnumbers', 'Ġto', 'Ġfind'
    ]

    # Generate a sample heatmap with random values for each token
    import random
    text_heatmap = [random.uniform(0.0, 1.0) for _ in text_tokens]

    # Combine tokens to form the raw text
    raw_text = "".join(text_tokens)

    # Test 1: Generate a basic image with wrapped text
    output_file_1 = "test_image_1.png"
    generator.generate_image(
        raw_text,
        output_file_1,
        force=True
    )
    print(f"Generated '{output_file_1}' without heatmap.\n")

    # Test 2: Generate an image with token-level heatmap
    output_file_2 = "test_image_2.png"
    generator.generate_image_withTextHeatmap(
        raw_text,
        output_file_2,
        text_tokens,
        text_heatmap,
        force=True
    )
    print(f"Generated '{output_file_2}' with token-level heatmap.\n")

    # generator.regenerate_image("test_image_1.png")
    # generator.subsample_image("test_image_1.png", 640000, "qwen")
    # generator.generate_image("Hello, this is a test.\n<img>\nThis should wrap and show random properties!",
    #                           "test_image_2.png", embeded_img_path="table_output.png")
