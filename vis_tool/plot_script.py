# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# def plot_two_charts_Sec33():
#     # -- 1) 数据准备 --
#     mmlu_dict = {
#         "GPT-4o": [38.5, 53.9, 41.1, 49.8],
#         "Claude 3.5 Sonnet": [42.7, 55.0, 43.9, 48.0],
#         "Gemini 1.5 Pro": [41.7, 49.4, 40.8, 44.5],
#         "GPT-4o mini": [29.8, 39.8, 30.0, 35.3],
#         "InternVL2-76B": [37.7, 41.9, 26.2, 38.0],
#         "LLaVA-OneVision-72B": [38.2, 38.0, 25.4, 24.0],
#         "ViLA1.5-40B": [35.2, 31.7, 15.6, 15.9],
#         "InternVL2-8B": [32.4, 32.5, 19.6, 25.4],
#         "Idefics3-8B": [29.7, 30.1, 13.9, 16.6]
#     }
#
#     mathverse_dict = {
#         "GPT-4V": [63.1, 54.7, 50.3, 31.6],
#         "QwenVL-Max": [42.8, 30.7, 35.9, 21.4],
#         "Gemini-Pro": [39.8, 26.3, 33.3, 22.2],
#         "LLaVA-NeXT-34B": [49.0, 33.8, 22.4, 15.7],
#         "XComposer2-7B": [36.9, 22.3, 19.8, 11.0],
#         "SPHINX-MoE": [33.3, 22.2, 18.3, 9.1],
#         "QwenVL-Plus": [26.0, 15.7, 21.8, 10.0],
#         "ShareGPT4V-13B": [21.8, 16.2, 9.7, 3.7],
#         "LLaVA-NeXT-13B": [21.6, 19.4, 12.1, 11.3],
#     }
#
#     mathverse_models = list(mathverse_dict.keys())
#     mmlu_models = list(mmlu_dict.keys())
#
#     sns.set_theme(style="white", font_scale=1.2)
#
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=False)
#
#     cot_color = "#F2A14B"  # 修改为浅黄色
#     bar_height = 0.25
#
#     # ========== 图1: MathVerse ==========
#     ax1 = axes[0]
#     text_cot_mv = [mathverse_dict[m][0] for m in mathverse_models]  # index=0
#     vision_cot_mv = [mathverse_dict[m][2] for m in mathverse_models]  # index=2
#     y_pos_mv = np.arange(len(mathverse_models))
#
#     bars1 = ax1.barh(
#         y_pos_mv - bar_height/2, text_cot_mv,
#         bar_height, label="Text-CoT", color=cot_color, alpha=0.9
#     )
#     bars2 = ax1.barh(
#         y_pos_mv + bar_height/2, vision_cot_mv,
#         bar_height, label="Vision-CoT", color=cot_color, alpha=0.5
#     )
#     ax1.set_yticks(y_pos_mv)
#     ax1.set_yticklabels(mathverse_models, fontsize=11)
#     ax1.invert_yaxis()
#     ax1.set_xlabel("Score", fontsize=12)
#     ax1.set_title("MathVerse", fontsize=15)
#     ax1.legend(fontsize=11)
#     ax1.bar_label(bars1, fmt='%.1f', label_type='edge', fontsize=10)
#     ax1.bar_label(bars2, fmt='%.1f', label_type='edge', fontsize=10)
#     ax1.grid(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['top'].set_visible(False)
#
#     # ========== 图2: MMMU ==========
#     ax2 = axes[1]
#     text_cot_mmlu = [mmlu_dict[m][0] for m in mmlu_models]  # index=1
#     vision_cot_mmlu = [mmlu_dict[m][2] for m in mmlu_models]  # index=3
#     y_pos_mmlu = np.arange(len(mmlu_models))
#
#     bars3 = ax2.barh(
#         y_pos_mmlu - bar_height/2, text_cot_mmlu,
#         bar_height, label="Text", color="#D65A31", alpha=0.9
#     )
#     bars4 = ax2.barh(
#         y_pos_mmlu + bar_height/2, vision_cot_mmlu,
#         bar_height, label="Vision", color="#D65A31", alpha=0.5
#     )
#     ax2.set_yticks(y_pos_mmlu)
#     ax2.set_yticklabels(mmlu_models, fontsize=11)
#     ax2.invert_yaxis()
#     ax2.set_xlabel("Score", fontsize=12)
#     ax2.set_title("MMMU", fontsize=15)
#     ax2.legend(fontsize=11)
#     ax2.bar_label(bars3, fmt='%.1f', label_type='edge', fontsize=10)
#     ax2.bar_label(bars4, fmt='%.1f', label_type='edge', fontsize=10)
#     ax2.grid(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#
#     max_x_value = max(
#         max([max(values) for values in mmlu_dict.values()]),
#         max([max(values) for values in mathverse_dict.values()])
#     )
#
#     ax1.set_xlim(0, max_x_value)
#     ax2.set_xlim(0, max_x_value)
#
#     plt.tight_layout()
#     plt.savefig("plot_two_charts.png", dpi=300, bbox_inches="tight")
#     plt.close()
#     print("绘图完成，结果已保存至 plot_two_charts.png")
#
# def plot_TableBench_Sec32():
#     pass
#
# def plot_All_Sec31():
#     pass
#
# if __name__ == "__main__":
#     plot_two_charts_Sec33()

import matplotlib.pyplot as plt
import seaborn as sns
def plot_figure_horizontal(result_dict,
                          output_path="plt.png",
                          model_order=None,
                          dataset_order=None,
                          model_name_map=None,
                          ncol=2):
    """
    Generates a grouped bar plot (horizontal) based on the result_dict and saves it to the specified output path.
    Adapts spacing to ensure readability.

    Parameters:
        result_dict (dict):
            A dictionary with keys as (model, dataset, mode) and values as numerical results.
            e.g. {("GPT4", "DatasetA", "text"): 90, ("GPT4", "DatasetA", "img"): 85, ...}
        output_path (str):
            File path to save the plot.
        model_order (list):
            Ordered list of models to display on y-axis.
        dataset_order (list):
            Ordered list of datasets (subgroup order) to display for each model.
        model_name_map (dict):
            Mapping of model names for display.
        ncol (int):
            Number of columns in the legend.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # === 1) 解析并排序维度 ===
    all_models = sorted(set([model for (model, _, _) in result_dict.keys()]))
    all_datasets = sorted(set([dataset for (_, dataset, _) in result_dict.keys()]))
    all_modes = sorted(set([mode for (_, _, mode) in result_dict.keys()]))

    if model_order:
        models = model_order
    else:
        models = all_models

    if dataset_order:
        datasets = dataset_order
    else:
        datasets = all_datasets

    # 调整 mode 的顺序：确保 img 在 text 左侧
    if set(all_modes) == {"text", "img"}:
        modes = ["img", "text"]  # 交换顺序，使 img 在左
    else:
        modes = sorted(all_modes)

    if model_name_map is None:
        model_name_map = {m: m for m in models}

    # === 2) 调整样式 + figure 基本设置 ===
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))  # 宽度大一些，因为是横向的

    # 只显示 y 方向的辅助线（水平线），不显示 x 网格
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.grid(axis='y', visible=False)

    # === 3) 计算每一组（model）与子组（dataset、mode）在 y 轴的位置 ===
    group_height = 0.8
    inter_model_gap = 0.01       # 不同 model 之间的额外间距（竖向）
    inter_dataset_gap = 0.005    # 同一 model 内部，不同 dataset 之间的空隙
    num_datasets = len(datasets)
    num_modes = len(modes)

    per_dataset_height = (group_height - (num_datasets - 1) * inter_dataset_gap) / num_datasets
    per_mode_height = per_dataset_height / num_modes

    base_colors = sns.color_palette("Set2", n_colors=num_datasets)
    legend_map = {}

    # === 4) 逐个绘制水平条 ===
    for i, model in enumerate(models):
        group_top = i - group_height / 2

        for j, dataset in enumerate(datasets):
            dataset_top = group_top + j * (per_dataset_height + inter_dataset_gap)
            base_color = base_colors[j]

            for k, mode in enumerate(modes):  # 现在 img 在左，text 在右
                y = dataset_top + k * per_mode_height
                value = result_dict.get((model, dataset, mode), 0.0)

                factor = 0.8 + 0.4 * (k / max(num_modes - 1, 1))
                color = adjust_color(base_color, factor)

                rects = ax.barh(y, value, height=per_mode_height, color=color, edgecolor="none")

                legend_key = (dataset, mode)
                if legend_key not in legend_map:
                    legend_label = f"{dataset.replace('Dataset', '')}({mode})"
                    legend_map[legend_key] = (rects, legend_label)

    # === 5) 设置坐标轴、刻度与标签 ===
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_name_map.get(m, m) for m in models], fontsize=12)
    ax.set_xlabel("Result", fontsize=13)
    # ax.set_ylabel("Models", fontsize=13)

    ax.set_xlim(0, 70)  # 根据数据可适当调整

    # === 6) 为每个条形添加数值注释 ===
    for rect in ax.patches:
        width = rect.get_width()
        y_bottom = rect.get_y()
        bar_height = rect.get_height()

        if width > 0:
            ax.text(
                width + 1.0,
                y_bottom + bar_height / 2,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontsize=10
            )

    # === 7) 设置图例 ===
    values = list(legend_map.values())
    values.reverse()  # 保证顺序与绘图时一致
    handles, labels = zip(*[(container[0], label) for container, label in values])
    ax.legend(
        handles,
        labels,
        title="Dataset(Mode)",
        loc="upper left",
        bbox_to_anchor=(0.75, 0.95),
        borderaxespad=0,
        ncol=ncol,
        fontsize=10,
        title_fontsize=12
    )

    # === 8) 布局与保存 ===
    plt.title("Models Comparison (Horizontal Grouped Bar)", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")



def adjust_color(rgb, factor=1.0):
    """
    简单的颜色明度调节函数:
      factor > 1 => 变亮
      factor < 1 => 变暗
    """
    r, g, b = rgb
    if factor > 1:
        return (min(r + (1 - r) * (factor - 1), 1),
                min(g + (1 - g) * (factor - 1), 1),
                min(b + (1 - b) * (factor - 1), 1))
    else:
        return (r * factor, g * factor, b * factor)

def plot_mmlu_only():
    """
    1) 只保留原先的 MMLU 数据
    2) 利用与 plot_figure 类似的分组柱状图格式进行绘制
    """

    # -- 1) 数据准备（仅保留 MMLU）--
    mmlu_dict = {
        "GPT-4o": [38.5, 53.9, 41.1, 49.8],
        "Claude 3.5 Sonnet": [42.7, 55.0, 43.9, 48.0],
        "Gemini 1.5 Pro": [41.7, 49.4, 40.8, 44.5],
        "GPT-4o mini": [29.8, 39.8, 30.0, 35.3],
        "InternVL2-76B": [37.7, 41.9, 26.2, 38.0],
        "LLaVA-OneVision-72B": [38.2, 38.0, 25.4, 24.0],
        "ViLA1.5-40B": [35.2, 31.7, 15.6, 15.9],
        "InternVL2-8B": [32.4, 32.5, 19.6, 25.4],
        "Idefics3-8B": [29.7, 30.1, 13.9, 16.6]
    }
    # 上面每个 list 有 4 个数值，这里仅示例地取 index=0 作为 text 模式、index=2 作为 vision 模式
    # 如果你需要使用 [1] 和 [3]，可以自行修改

    # -- 2) 转成与 plot_figure() 相适配的 result_dict 格式 --
    # 假设想把这张表都叫做 "MMMU" 这个 dataset，mode 分别是 "text" 和 "img"
    result_dict = {}
    for model, values in mmlu_dict.items():
        # 以第一个值当做 text，第三个值当做 vision
        text_score = values[0]
        vision_score = values[2]
        result_dict[(model, "MMMU", "text")] = text_score
        result_dict[(model, "MMMU", "img")] = vision_score

    # 这里按原 dict 的顺序来排序模型，也可以根据需要自定义
    model_order = list(mmlu_dict.keys())
    model_order.reverse()

    # -- 3) 调用 plot_figure() 函数绘图 --
    # dataset_order 只有一个 "MMMU"，mode 自动会识别为 ["img", "text"]，
    # 但是我们想让它顺序为 ["text", "img"]，可以在后面手动调整
    plot_figure_horizontal(
        result_dict=result_dict,
        output_path="plot_two_charts.png",
        model_order=model_order,
        dataset_order=["MMMU"],    # 只有一个数据集
        model_name_map=None,      # 如果需要对模型名做映射可以传入字典
        ncol=1                   # 只显示一个数据集 + 2 种 mode，所以两列即可
    )

    print("MMLU 绘图完成，结果已保存至 plot_two_charts.png")

# 你可以直接调用 plot_mmlu_only() 来生成图
if __name__ == "__main__":
    plot_mmlu_only()