import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch

def build_image_token_mask(inputs):
    """
    根据输入的 pixel_values 重新计算哪些 Patch 被删除，然后把 outputs.attentions 对应被删除的列和行补回去 (填 0)。
    这样就可以在后续可视化时，继续用 (height, width) = (14, 14) 来 reshape。

    参数：
    - inputs: 一个包含 pixel_values, image_grid_thw 等的字典
    - outputs: model(...) 的输出 (dict)，包含 'logits' 和 'attentions'

    返回：
    - 一个新的 outputs，对 attentions 进行了“还原”操作
    """

    # --------------------
    # 1) 重新跑一次 PatchEmbed 的逻辑，获取 white_space_mask
    # --------------------
    pixel_values = inputs["pixel_values"]
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 14
    num_pixels_per_patch = in_channels * temporal_patch_size * patch_size * patch_size

    assert pixel_values.numel() % num_pixels_per_patch == 0, \
        "pixel_values 的元素总数不符合 (C*T*H*W) 整除关系，请检查。"

    num_patches = pixel_values.numel() // num_pixels_per_patch
    patches = pixel_values.view(num_patches, in_channels, temporal_patch_size, patch_size, patch_size)

    # 1.2) 和 PatchEmbed 里一样，做 group of 4
    #     这里演示：假设确实是 batch=1，但为了保持和你给的 PatchEmbed 一致，这里也凑成了 4 的倍数
    num_groups_of_4 = num_patches // 4
    patches_group4 = patches.view(num_groups_of_4, 4, in_channels, temporal_patch_size, patch_size, patch_size)

    # 1.3) 计算方差
    patches_group4_flat = patches_group4.view(
        num_groups_of_4,
        4,
        in_channels,
        temporal_patch_size * patch_size * patch_size
    )
    patch_variances = patches_group4_flat.var(dim=-1)  # [num_groups_of_4, 4, C]
    patch_variances_mean = patch_variances.mean(dim=-1)  # [num_groups_of_4, 4]
    patch_variances_mean_max = patch_variances_mean.max(dim=-1).values  # [num_groups_of_4,]

    variance_threshold = 0.01
    white_space_mask_group4 = (patch_variances_mean_max > variance_threshold)  # [num_groups_of_4]

    if not white_space_mask_group4.any():
        white_space_mask_group4[0] = True
    return white_space_mask_group4

def greedy_decode_with_attention_img(
        model,
        inputs,
        processor,
        max_new_tokens=100,
        image_token_id=151655,
        height=14,
        width=14,
        layer_index=-1,
        lighting_mode = True
):
    """
    使用贪心解码 (greedy decoding)，在每一步生成token时，都输出当前这一步的注意力 (文本->图像patch)。
    这里假设 `inputs` 是已经用 processor(text=..., images=..., return_tensors="pt") 编码完的结果。

    参数：
    - model: Qwen2VLForConditionalGeneration 或类似的多模态模型
    - inputs: 由 processor 编码后的输入，必须包含 input_ids, attention_mask (以及 pixel_values, 如果需要)
    - processor: 处理器(仅在本示例中，如果你要做 decode 结束后将 token 转成文本、可视化等，可能会用到)
    - max_new_tokens: 生成的步数上限
    - image_token_id: 对应图像 token 的 ID
    - height, width: 重排成 Heatmap 的形状
    - layer_index: 指定要获取的 Layer 索引 (如果是 -1 则表示最后一层)

    返回:
    - generated_tokens: 生成的 token 序列（Python int list）
    - attentions_per_step: 长度与生成步数相同的列表, 每个元素是一个 heatmap (形状: `[height, width]`)
    """
    device = next(model.parameters()).device
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)

    input_ids = inputs["input_ids"]  # [1, seq_len]
    attention_mask = inputs["attention_mask"]  # [1, seq_len]
    pixel_values = inputs["pixel_values"] if "pixel_values" in inputs else None

    generated_tokens = input_ids[0].tolist()  # 转成 Python list
    attentions_per_step = []

    for step in range(max_new_tokens):
        curr_input_ids = torch.tensor([generated_tokens], device=device)
        curr_attention_mask = torch.ones_like(curr_input_ids, device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                attention_mask=curr_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=inputs['image_grid_thw'],
                output_attentions=True,
                return_dict=True
            )

        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_token_id = logits.argmax(dim=-1).item()  # 贪心选
        generated_tokens.append(next_token_id)

        all_attentions = outputs.attentions
        sel_layer_idx = len(all_attentions) + layer_index if layer_index < 0 else layer_index
        layer_attention = all_attentions[sel_layer_idx]

        all_ids = torch.tensor(generated_tokens, device=device)
        image_positions = (all_ids == image_token_id).nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"未在 input_ids 中找到图像 token_id={image_token_id}，请检查！")
        image_start_idx = int(image_positions[0].item())

        if lighting_mode:
            masks = build_image_token_mask(inputs)
            valid_image_tokens = masks.sum().item()
            new_image_end_idx = image_start_idx + valid_image_tokens

            # print(image_start_idx, new_image_end_idx)
            # print(len(masks), sum(masks))
            # exit(0)

            if new_image_end_idx > len(masks):
                raise ValueError(
                    f"新的 image_end_idx ({new_image_end_idx}) 超出范围 len(masks) ({len(masks)})，请检查 masks 和输入！"
                )

            bsz, num_heads, seq_len_from_attn, _ = layer_attention.shape
            if new_image_end_idx > seq_len_from_attn:
                raise ValueError(
                    f"Invalid image token range [{image_start_idx}, {new_image_end_idx}). "
                    "Check your special token IDs or the model's tokenization process."
                )

            query_idx = outputs.logits.shape[1] - 1  # 锁定为 output 的最后一位
            token_image_attention = layer_attention[:, :, query_idx, image_start_idx:new_image_end_idx]
            token_image_attention_avg = token_image_attention.abs().mean(dim=1)

            heatmap_expanded = torch.zeros(len(masks), device=device, dtype=token_image_attention_avg.dtype)
            heatmap_expanded[masks.bool()] = torch.tensor(token_image_attention_avg.flatten(), device=device)
            heatmap_resized = heatmap_expanded.view(height, width).to(torch.float32).detach().cpu().numpy()
            attentions_per_step.append(heatmap_resized)
        else:
            image_end_idx = int(image_positions[-1].item()) + 1

            bsz, num_heads, seq_len_from_attn, _ = layer_attention.shape
            if image_end_idx > seq_len_from_attn:
                raise ValueError(
                    f"Invalid image token range [{image_start_idx}, {image_end_idx}). "
                    "Check your special token IDs or the model's tokenization process."
                )

            query_idx = outputs.logits.shape[1] - 1  # 锁定为 output 的最后一位
            token_image_attention = layer_attention[:, :, query_idx, image_start_idx:image_end_idx]
            token_image_attention_avg = token_image_attention.abs().mean(dim=1)

            heatmap = token_image_attention_avg.view(1, height, width)
            heatmap_2d = heatmap[0].to(torch.float32).detach().cpu().numpy()
            attentions_per_step.append(heatmap_2d)

        if next_token_id == model.config.eos_token_id:
            del outputs
            torch.cuda.empty_cache()
            break

        del outputs
        torch.cuda.empty_cache()

    return generated_tokens, attentions_per_step


def greedy_decode_with_attention_text(
    model,
    inputs,
    processor,
    max_new_tokens=100,
    image_token_id=151655,
    layer_index=-1
):
    """
    贪心解码（文本版），并仅在每一步取对“用户问题那段 token”的注意力。

    返回:
    - question_tokens: 用户问题对应的 token 列表（已在第二个 <|im_start|> 后跳过两个 token）
    - answer_tokens: 模型新生成的 token 列表（**不**含 system/user 块）
    - attentions_per_step: 每一步生成时，新 token 对“用户问题”那段的注意力分布 (list of numpy arrays)
    """

    device = next(model.parameters()).device

    # Move all input tensors to device
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)

    # Full input (including system + user blocks)
    full_input_ids = inputs["input_ids"]  # shape: [1, seq_len]
    attention_mask = inputs["attention_mask"]  # [1, seq_len]
    pixel_values = inputs.get("pixel_values", None)  # ignored in text scenario

    # ---------------------------
    # 1. Identify the user block
    # ---------------------------
    start_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    input_ids_list = full_input_ids[0].tolist()

    # Find all occurrences of <|im_start|> and <|im_end|>
    start_positions = [i for i, tok in enumerate(input_ids_list) if tok == start_token_id]
    end_positions = [i for i, tok in enumerate(input_ids_list) if tok == end_token_id]

    if len(start_positions) < 2 or len(end_positions) < 2:
        raise ValueError("Could not find second <|im_start|> or second <|im_end|> in input_ids. "
                         "Check that your input truly contains system+user blocks.")

    # Typical Qwen usage:
    #   1st <|im_start|> => system
    #   1st <|im_end|>   => system end
    #   2nd <|im_start|> => user
    #   2nd <|im_end|>   => user end
    user_block_start = start_positions[1]  # 2nd <|im_start|>
    user_block_end_candidates = [pos for pos in end_positions if pos > user_block_start]
    if not user_block_end_candidates:
        raise ValueError("Could not find a <|im_end|> after the second <|im_start|> (user block).")
    user_block_end = user_block_end_candidates[0]

    # --------------------------------
    # 跳过第二个 <|im_start|> 后的两个 token
    # --------------------------------
    # 即从 user_block_start + 3 开始当作用户真正的问题内容
    # （如果要确保序列不越界，可以做下检查）
    user_content_start = user_block_start + 3
    user_content_end = user_block_end  # exclusive
    if user_content_start >= user_content_end:
        raise ValueError("User content block is empty or invalid after skipping 2 tokens.")

    # 取出用户问题对应的那段 ID
    user_input_ids_slice = input_ids_list[user_content_start:user_content_end]
    if len(user_input_ids_slice) <= 0:
        raise ValueError("No user tokens found in user block. Possibly your user message is empty?")

    # 转成 tokens => question_tokens
    question_tokens = processor.tokenizer.convert_ids_to_tokens(user_input_ids_slice)

    # ------------------------------------------------
    # 2. 开始贪心解码并记录注意力
    # ------------------------------------------------
    generated_tokens = input_ids_list[:]  # 从整段 prompt (system + user) 开始
    attentions_per_step = []

    for step in range(max_new_tokens):
        curr_input_ids = torch.tensor([generated_tokens], device=device)
        curr_attention_mask = torch.ones_like(curr_input_ids, device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                attention_mask=curr_attention_mask,
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        # 取最后一步的 logits
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_token_id = logits.argmax(dim=-1).item()
        generated_tokens.append(next_token_id)

        # 取指定层的注意力
        all_attentions = outputs.attentions
        sel_layer_idx = len(all_attentions) + layer_index if layer_index < 0 else layer_index
        layer_attention = all_attentions[sel_layer_idx]  # [1, num_heads, seq_len, seq_len]

        # 新生成 token 对应的 query index
        query_idx = layer_attention.shape[2] - 1  # 序列最后一个
        # shape: [1, num_heads, seq_len]
        token_attention = layer_attention[:, :, query_idx, :]

        # 对所有头做平均 => [seq_len]
        token_attention_avg = token_attention.mean(dim=1).squeeze(0)

        # 只截取“用户问题”那段
        seq_len_now = token_attention_avg.shape[0]
        user_content_end_clamped = min(user_content_end, seq_len_now)
        if user_content_start < seq_len_now:
            user_portion_attention = token_attention_avg[user_content_start:user_content_end_clamped]
        else:
            user_portion_attention = torch.zeros(0, device=device)

        # 放到 CPU numpy
        user_portion_attention_np = user_portion_attention.detach().cpu().float().numpy()
        attentions_per_step.append(user_portion_attention_np)

        # 检测 EOS
        if next_token_id == model.config.eos_token_id:
            del outputs
            torch.cuda.empty_cache()
            break

        del outputs
        torch.cuda.empty_cache()

    # ------------------------------------------------
    # 3. 取最后输出 token（answer_tokens）
    #    （跳过原本 prompt 的所有 token）
    # ------------------------------------------------
    prompt_len = len(input_ids_list)
    new_tokens = generated_tokens[prompt_len:]
    answer_tokens = processor.tokenizer.convert_ids_to_tokens(new_tokens)

    return question_tokens, answer_tokens, attentions_per_step


def test_visionHeatmap():
    from analyze_utils import load_model_and_input_withImg
    model, inputs, processor, img_path = load_model_and_input_withImg(img_path=f"vis_result/MBPP_120_pure.png",
                                                                      text=f"Please answer the question in the image.")
    print("inputs:", inputs)
    print("image_grid_thw:", inputs['image_grid_thw'])
    # exit(0)
    token_index = 1  # 第二个 token
    layer_index = -1  # 最后一层
    # layer_index = 0  # 最后一层

    generated_tokens, attentions_per_step = greedy_decode_with_attention_img(model, inputs, processor, layer_index = layer_index,
                                              height = inputs['image_grid_thw'][0,1]//2,
                                              width = inputs['image_grid_thw'][0,2]//2)

    # print("Generated tokens:", generated_tokens)
    # 假设你在上面已经获得了 generated_tokens
    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Decoded text:", decoded_text)

    print("Length of attentions_per_step:", len(attentions_per_step))
    print(attentions_per_step[0].shape)
    print(attentions_per_step[0])

    output_dir = "plot_cache"
    os.makedirs(output_dir, exist_ok=True)

    from patch_visualizer import visualize_image_patches

    # Visualize each step's heatmap and save
    for i, attention in enumerate(attentions_per_step):
        heatmap_path = os.path.join(output_dir, f"heatmap_step_{i + 1}.png")
        visualize_image_patches(
            image_path=img_path,  # Replace with your input image path
            output_path=heatmap_path,
            patch_size=28,
            heatmap=attention
        )
    # Calculate the average attention across all steps
    average_attention = sum(attentions_per_step) / len(attentions_per_step)

    # Save the "All Average" heatmap
    all_average_heatmap_path = os.path.join(output_dir, "heatmap_all_average.png")
    visualize_image_patches(
        image_path=img_path,  # Replace with your input image path
        output_path=all_average_heatmap_path,
        patch_size=28,
        heatmap=average_attention
    )

    print(f"Heatmaps have been saved to the '{output_dir}' directory.")


def test_textHeatmap():
    from analyze_utils import load_model_and_input_withText
    model, inputs, processor = load_model_and_input_withText()

    # Call your text-only greedy decoding function
    # (Make sure you have already defined greedy_decode_with_attention_text
    #  similarly to greedy_decode_with_attention_img.)
    question_tokens, text_tokens, attentions_per_step = greedy_decode_with_attention_text(
        model,
        inputs,
        processor,
        max_new_tokens=30,  # For example, generate up to 30 new tokens
        image_token_id=151655  # Not used for text-only but kept for consistency
    )

    print(text_tokens)
    # Convert IDs to text string
    # decoded_text = processor.tokenizer.decode(text_tokens, skip_special_tokens=True)
    print("Decoded text:", text_tokens)

    # Print out attention stats
    print(f"\nNumber of steps: {len(attentions_per_step)}")
    for i, attention in enumerate(attentions_per_step):
        print(f"[Step {i + 1}] attention shape: {attention.shape}, sum: {attention.sum():.4f}")

    # If we have at least one attention map, compute an average
    average_attention = sum(attentions_per_step) / len(attentions_per_step) if attentions_per_step else np.zeros(0)
    # Normalize to [0, 1] for visualization
    average_attention = (average_attention - average_attention.min()) / (average_attention.max() -
                                                                         average_attention.min()) if attentions_per_step else np.zeros(0)

    from image_generate import ImageGenerator_withTextHeatmap
    generator = ImageGenerator_withTextHeatmap(
        random_image_scale=True,
        random_font_size=True,
        random_padding=True,
        random_font=True,
        add_low_freq_noise=True,
        add_high_freq_noise=True,
        no_random=True,
    )
    raw_text = "".join(question_tokens)

    output_file_1 = "heatmap_image_1.png"
    generator.generate_image(
        raw_text,
        output_file_1,
        force = True
    )
    print(f"Generated '{output_file_1}' without heatmap.\n")

    output_file_2 = "heatmap_image_2.png"
    generator.generate_image_withTextHeatmap(
        raw_text,
        output_file_2,
        question_tokens,
        average_attention,
        force=True
    )
    print(f"Generated '{output_file_2}' withheatmap.\n")
    print("\n=== Done: text_textHeatmap ===")

def visHeatmap_pipeline(pure_text_input, output_name):
    from image_generate import ImageGenerator_withTextHeatmap
    generator = ImageGenerator_withTextHeatmap(
        random_image_scale=True,
        random_font_size=True,
        random_padding=True,
        random_font=True,
        add_low_freq_noise=True,
        add_high_freq_noise=True,
        no_random=True,
    )
    from analyze_utils import load_model_and_input_withImg, load_model_and_input_withText
    model, inputs, processor = load_model_and_input_withText(text=pure_text_input)
    question_tokens, text_tokens, attentions_per_step = greedy_decode_with_attention_text(
        model,
        inputs,
        processor,
        max_new_tokens=512,  # For example, generate up to 30 new tokens
        image_token_id=151655  # Not used for text-only but kept for consistency
    )
    # decoded_text = processor.tokenizer.decode(text_tokens, skip_special_tokens=True)
    print("Decoded text in Text:", generator.process_text_fromDecoder("".join(text_tokens)))

    average_attention = sum(attentions_per_step) / len(attentions_per_step) if attentions_per_step else np.zeros(0)
    # Normalize to [0, 1] for visualization
    average_attention = (average_attention - average_attention.min()) / (average_attention.max() -
                                                                         average_attention.min()) if attentions_per_step else np.zeros(0)
    output_file_1, output_file_2 = f"vis_result/{output_name}_pure.png", f"vis_result/{output_name}_text.png"
    raw_text = "".join(question_tokens)
    generator.generate_image(
        pure_text_input,
        output_file_1,
        force=True
    )
    generator.generate_image_withTextHeatmap(
        pure_text_input,
        output_file_2,
        question_tokens,
        average_attention,
        force=True
    )
    print("\n=== Done: text heatmap ===")

    del model
    torch.cuda.empty_cache()

    model, inputs, processor, img_path = load_model_and_input_withImg(img_path=output_file_1,
                                                                      text=f"Please answer the question in the image.")
    generated_tokens, attentions_per_step = greedy_decode_with_attention_img(model, inputs, processor,
                                              height = inputs['image_grid_thw'][0,1]//2,
                                              width = inputs['image_grid_thw'][0,2]//2,
                                              max_new_tokens = 512,
                                            )

    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Decoded text in Img:", decoded_text)

    from patch_visualizer import visualize_image_patches
    average_attention = sum(attentions_per_step) / len(attentions_per_step)

    # Save the "All Average" heatmap
    all_average_heatmap_path = f"vis_result/{output_name}_img.png"
    visualize_image_patches(
        image_path=img_path,  # Replace with your input image path
        output_path=all_average_heatmap_path,
        patch_size=28,
        heatmap=average_attention
    )
    print("\n=== Done: img heatmap ===")


if __name__ == "__main__":
    # test_visionHeatmap()
#     example_text = """Below is a Python problem:
#
# To validate your solution, we will run these tests:
# assert insert_element(['Red', 'Green', 'Black'], 'c')==['c', 'Red', 'c', 'Green', 'c', 'Black']
# assert insert_element(['python', 'java'], 'program')==['program', 'python', 'program', 'java']
# assert insert_element(['happy', 'sad'], 'laugh')==['laugh', 'happy', 'laugh', 'sad']
# Please write a Python function that solves the above problem.
# Please only provide your final answer after 'Answer:'.
# Please use the following template:
# Answer:[TODO]"""
#     visHeatmap_pipeline(example_text, "MBPP_120_light")
    example_text="""Task: BoolQ\nThe BoolQ task requires the model to answer a yes/no question based on a given passage.\nQuestion: will there be a sequel to the movie predators\nPassage: Predators (film) -- Predators is a 2010 American science-fiction action film directed by Nimród Antal and starring Adrien Brody, Topher Grace, Alice Braga, Walton Goggins, and Laurence Fishburne. It was distributed by 20th Century Fox. It is the third installment of the Predator franchise (the fifth counting the two Alien vs. Predator films), following Predator (1987) and Predator 2 (1990). Another film, The Predator, is set for release in 2018.\nYour final answer must be exactly 'True' or 'False'.\nPlease only provide your final answer after 'Answer:'.\nPlease use the following template:\nAnswer:[TODO]"""
    visHeatmap_pipeline(example_text, "BoolQ_1113_light")


