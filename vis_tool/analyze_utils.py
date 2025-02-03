import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_model_and_input_withImg(img_path=None, text=None):
    """
    Load the Qwen2-VL-7B-Instruct model and prepare the input data.
    Returns:
        model: The loaded Qwen2VLForConditionalGeneration model.
        inputs: Preprocessed inputs ready for the model.
        processor: The processor used for input preparation.
    """
    # Load the model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Use the provided img_path and text if available
    if img_path is None:
        img_path = "/home/zhiheng/WordAsPixel/image_cache/ARC/base/ARC-Challenge/0.png"

    if text is None:
        text = "Please follow the instruction in the image."

    # Create input message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": text},
            ],
        }
    ]

    # Process input text and images
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Ensure image inputs are tensors
    if image_inputs:
        image_tensors = [
            processor.image_processor(image, return_tensors="pt")["pixel_values"]
            for image in image_inputs
        ]
        print(f"Image Tensors Shape: {[img.shape for img in image_tensors]}")

    # Prepare inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    return model, inputs, processor, img_path

def load_model_and_input_withText(text=None):
    """
    Load the Qwen2-VL-7B-Instruct model and prepare a multiple-choice math question input.

    Returns:
        model: The loaded Qwen2VLForConditionalGeneration model.
        inputs: The preprocessed text inputs (ready for model inference).
        processor: The processor used for input preparation.
    """
    # Load the model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Use the provided text if available
    if text is None:
        text = (
            "Here is a math question:\n"
            "3 + 5 = ?\n"
            "Options:\n"
            "A) 5\n"
            "B) 7\n"
            "C) 8\n"
            "D) 9\n\n"
            "Please select the correct option and explain your reasoning."
        )

    # Create a user message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        }
    ]

    # Convert the messages to a single string with the Qwen chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare inputs for the model (text only, no images/videos)
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    # Move everything to GPU if available
    inputs = inputs.to("cuda")

    return model, inputs, processor

def visualize_model_and_inputs(model, inputs, processor):
    """
    Visualize model parameters and processed input data.
    Args:
        model: The loaded Qwen2VLForConditionalGeneration model.
        inputs: Preprocessed inputs ready for the model.
        processor: The processor used for decoding outputs.
    """
    # Print model parameters
    def print_model_parameters(model):
        total_params = 0
        print("Model Parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
            total_params += param.numel()
        print(f"Total Parameters: {total_params:,}")

    print_model_parameters(model)

    # Visualize input shapes
    print("Text Input IDs Shape:", inputs.input_ids.shape)
    if "pixel_values" in inputs:
        print("Image Pixel Values Shape:", inputs.pixel_values.shape)

    # Perform inference and display generated output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Generated Output Text:", output_text)

def get_global_average_embeddings(model, inputs):
    """
    获取模型最后一层（文本部分或文本+图像融合部分）的全局平均嵌入向量。

    参数:
        model: Qwen2VLForConditionalGeneration (或其他兼容的多模态模型)
        inputs: 经过 processor(...) 后的张量输入, 包含 input_ids, pixel_values 等

    返回:
        global_average_embeddings: (batch_size, hidden_size) 的张量，表示每个样本的全局平均嵌入
    """

    # 推理时输出 hidden_states，需要在 forward 时传入 output_hidden_states=True
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        # 获取最后一层隐状态
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        # 计算全局平均
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_hidden_states = last_hidden_states * attention_mask  # 忽略 padding 的部分
        sum_hidden_states = masked_hidden_states.sum(dim=1)  # 对序列维度求和 (batch_size, hidden_size)
        valid_token_count = attention_mask.sum(dim=1)  # 每个样本中有效 token 的数量 (batch_size, 1)
        global_average_embeddings = sum_hidden_states / valid_token_count  # 平均值 (batch_size, hidden_size)

    return global_average_embeddings

# 示例调用
# check_token_consistency(inputs)

if __name__ == "__main__":
    # Load model, inputs, and processor
    model, inputs, processor = load_model_and_input_withImg()
    # print(inputs['pixel_values'].shape)
    # check_token_consistency(inputs)

    # Visualize model parameters and inputs
    # visualize_model_and_inputs(model, inputs, processor)

    # 获取最后一层的嵌入向量
    # last_hidden_states = get_global_average_embeddings(model, inputs)
    # print("Last Hidden States Shape:", last_hidden_states.shape)
    # exit(0)
    # 获取指定 Layer 的注意力 Heatmap
    token_index = 1  # 第二个 token
    layer_index = -1  # 最后一层

    # print(inputs)
    # exit(0)
    attention_heatmap = get_attention_heatmap(model, inputs, layer_index = layer_index,
                                              height = inputs['image_grid_thw'][0,1]//2,
                                              width = inputs['image_grid_thw'][0,2]//2)
    print("Attention Heatmap Shape:", attention_heatmap.shape)
    print(attention_heatmap.max(), attention_heatmap.min())

    # # 获取基于梯度的 Heatmap
    # gradient_heatmap = get_gradient_heatmap(model, inputs, token_index, layer_index)
    # print("Gradient Heatmap Shape:", gradient_heatmap.shape)


