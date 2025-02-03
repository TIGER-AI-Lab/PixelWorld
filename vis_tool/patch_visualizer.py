import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_image_patches(image_path, output_path, patch_size=28, heatmap=None):
    """
    Visualize an image divided into patches by resizing the image to the nearest
    multiple of `patch_size` (instead of padding), and optionally overlay a heatmap.
    """

    # 1. 读取并检查图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    if len(img.shape) == 2:  # 若是灰度图，则转 BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 原始尺寸
    orig_height, orig_width = img.shape[:2]

    # 2. 计算新的对齐尺寸 (使用四舍五入，但可根据需求修改成下取整或上取整)
    #    如果想要完全复刻 Qwen2VL 里更复杂的 smart_resize，可以自行改造。
    new_height = round(orig_height / patch_size) * patch_size
    new_width  = round(orig_width  / patch_size) * patch_size

    # 若 new_height 或 new_width 变成 0，需特殊处理
    new_height = max(new_height, patch_size)
    new_width  = max(new_width, patch_size)

    # 3. 使用插值方式，将原图缩放到 new_height x new_width
    #   插值方式可选 cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC 等
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 4. 计算网格行列数
    grid_h = new_height // patch_size
    grid_w = new_width  // patch_size

    print(f"Original image size: {orig_width}x{orig_height}")
    print(f"Resized image size:  {new_width}x{new_height}")
    print(f"Grid: {grid_w} columns x {grid_h} rows, Patch size: {patch_size}x{patch_size}")

    # 5. 处理 heatmap（可选）
    if heatmap is not None:
        # 检查 heatmap 大小是否匹配 (grid_h, grid_w)
        if heatmap.shape != (grid_h, grid_w):
            # 若恰好是 (grid_w, grid_h) 就做转置
            if heatmap.shape == (grid_w, grid_h):
                heatmap = heatmap.T
                print("Transposed heatmap to match (grid_h, grid_w).")
            else:
                raise ValueError(
                    f"Heatmap shape {heatmap.shape} does not match grid ({grid_h},{grid_w})."
                )

        # (a) 归一化到 [0, 255]
        heatmap_normalized = cv2.normalize(
            heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)

        # (b) 将 (grid_h, grid_w) -> (new_height, new_width) via resize
        heatmap_resized = cv2.resize(
            heatmap_normalized,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST  # 保持每个小块不被平滑
        )

        # (c) 应用彩色映射
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # (d) 叠加到图像上
        alpha = 0.3
        overlay = cv2.addWeighted(heatmap_color, alpha, img_resized, 1 - alpha, 0.0)
    else:
        overlay = img_resized.copy()

    # 6. 画网格线
    # 行
    for y in range(0, new_height, patch_size):
        cv2.line(overlay, (0, y), (new_width, y), color=(255, 255, 255), thickness=1)
    # 列
    for x in range(0, new_width, patch_size):
        cv2.line(overlay, (x, 0), (x, new_height), color=(255, 255, 255), thickness=1)

    # 7. 保存与可视化
    cv2.imwrite(output_path, overlay)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_rgb)
    plt.title("Image with Patch Grid (Resized, no Padding) and Heatmap")
    plt.axis("off")
    plt.show()

# Example usage

# def visualize_image_patchs_withTextHeatmap(image_path, output_path, patch_size=28, text_tokens=None, tokens_Heatmap):
#     pass


if __name__ == "__main__":
    input_image_path = "/home/zhiheng/WordAsPixel/image_cache/ARC/base/ARC-Challenge/0.png"  # Replace with the path to your image
    output_image_path = "output_image_with_grid.jpg"  # Replace with the desired output path

    patch_size = 28
    visualize_image_patches(input_image_path, output_image_path, patch_size=patch_size)
