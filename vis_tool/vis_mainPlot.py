import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import numpy as np

def extract_df(paths=["eval_cache.json"]):
    # Initialize an empty list to store the processed data
    data = []

    # Iterate over each JSON file path
    for path in paths:
        with open(path, 'r') as file:
            json_data = json.load(file)

        # Process each key-value pair in the JSON
        for key, value in json_data.items():
            model = value.get("model")
            dataset = value.get("dataset")
            mode = value.get("mode")
            prompt = value.get("prompt")

            # Extract task results
            task_results = value.get("task_results", {})
            for subset_name, result in task_results.items():
                data.append({
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "prompt": prompt,
                    "subset_name": subset_name,
                    "result": result
                })

            # Add the final_score entry
            final_score = value.get("final_score")
            if final_score is not None:
                data.append({
                    "model": model,
                    "dataset": dataset,
                    "mode": mode,
                    "prompt": prompt,
                    "subset_name": "final_score",
                    "result": final_score
                })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    # print(df)
    return df

def find_badcase(model = "Qwen2-VL-7B-Instruct", dataset = "MMLUProDataset"):
    def get_idx(dataDict_text, dataDict_img, class_name, datapoint_idx):
        print("#" * 10)
        print(dataDict_text[class_name][datapoint_idx])
        return dataDict_text[class_name][datapoint_idx], dataDict_img[class_name][datapoint_idx]
    # Dynamically import data class
    try:
        data_module = importlib.import_module('data')
        data_class = getattr(data_module, dataset)
    except (ModuleNotFoundError, AttributeError) as e:
        raise e
        raise ValueError(f"Model wrapper class '{dataset}' not found in 'data' module.")
    data_object = data_class()
    dataDict_text = data_object.PureTextData()
    dataDict_img = data_object.VisionData()
    answers = [i[1] for i in dataDict_text["biology"]]
    # print(answers)
    text_cache_path = f"result_cache/{model}/MMLU-Pro/biology_text_Base.json"
    img_cache_path = f"result_cache/{model}/MMLU-Pro/biology_img_Base.json"
    with open(text_cache_path, 'r') as file:
        text_cache = json.load(file)
    with open(img_cache_path, 'r') as file:
        img_cache = json.load(file)
    # print(data_dict.keys())
    # print(answers)
    # print(text_cache)
    # print(img_cache)
    text_result = [data_object.parse_output(None, i) for i in text_cache]
    img_result = [data_object.parse_output(None, i) for i in img_cache]
    answer_result = [data_object.parse_output(None, i) for i in answers]

    TP_idx, FP_text_idx, FP_img_idx, FN_idx = [], [], [], []
    for i in range(len(answers)):
        if text_result[i] == answer_result[i] and img_result[i] == answer_result[i]:
            TP_idx.append(i)
        elif text_result[i] != answer_result[i] and img_result[i] != answer_result[i]:
            FN_idx.append(i)
        elif text_result[i] != answer_result[i]:
            FP_text_idx.append(i)
        elif img_result[i] != answer_result[i]:
            FP_img_idx.append(i)
    print(f"TP: {len(TP_idx)}")
    print(f"FP_text: {len(FP_text_idx)}")
    print(f"FP_img: {len(FP_img_idx)}")
    print(f"FN: {len(FN_idx)}")
    print(f"TP example: {TP_idx}")
    print(f"FP_text example: {FP_text_idx}")
    print(f"FP_img example: {FP_img_idx}")
    print(f"FN example: {FN_idx}")
    tp_sample = get_idx(dataDict_text, dataDict_img, "biology", TP_idx[0])
    fp_text_sample = get_idx(dataDict_text, dataDict_img, "biology", FP_text_idx[0])
    fp_img_sample = get_idx(dataDict_text, dataDict_img, "biology", FP_img_idx[0])
    return tp_sample, fp_text_sample, fp_img_sample

# Legacy Code
def vis_single_barplot(paths=["eval_cache.json"], output_path="plt.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")  # 初始风格带网格

    df = extract_df(paths)

    # ---------------------------
    # 1. 一些映射及列表
    # ---------------------------
    dataset_prompt_map = {
        "SuperGLUEDataset":  "base",
        "GLUEDataset":       "base",
        "MMLUProDataset":    "cot",
        "ARCDataset":        "base",
        "MBPPDataset":       "base",
        "GSM8KDataset":      "cot",
        "TableBenchDataset": "base",  # 这个不 *100
    }

    dataset_list = [
        "SuperGLUEDataset", "GLUEDataset", "MMLUProDataset",
        "ARCDataset", "MBPPDataset", "GSM8KDataset", # "TableBenchDataset"
    ]

    model_order = ["Qwen2_VL_2B", "Phi_3_5_vision", "Qwen2_VL_7B", "Gemini_Flash", "GPT4o"]
    model_name_map = {
        "Qwen2_VL_2B":     "QWen2VL-2B",
        "Phi_3_5_vision":  "Phi-3.5-4.2B",
        "Qwen2_VL_7B":     "QWen2VL-7B",
        "Gemini_Flash":    "Gemini-Flash",
        "GPT4o":           "GPT4o",
        # "Gemini_Thinking":           "Gemini-Thinking",
    }

    # ---------------------------
    # 2. 过滤数据
    # ---------------------------
    df["prompt_ok"] = df["dataset"].map(dataset_prompt_map) == df["prompt"]
    df = df[df["dataset"].isin(dataset_list) & df["prompt_ok"]]
    df = df[df["model"].isin(model_order)]

    # 除 TableBenchDataset 之外都 *100
    mask_not_table = df["dataset"] != "TableBenchDataset"
    df.loc[mask_not_table, "result"] = df.loc[mask_not_table, "result"] * 100

    # 聚合
    df_agg = df.groupby(["model", "dataset", "mode"], as_index=False)["result"].mean()
    # 放进 dict 便于访问
    result_dict = {}
    for row in df_agg.itertuples():
        result_dict[(row.model, row.dataset, row.mode)] = row.result

    # ---------------------------
    # 3. 设置图表和自定义参数
    # ---------------------------
    fig, ax = plt.subplots(figsize=(16, 6))

    # 只保留水平网格线，移除垂直网格线
    # (先让白格子全开，再只留 y 轴方向的虚线)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)

    x_positions = range(len(model_order))
    group_width = 0.8
    # 1) 你想让 dataset 间的 gap 比之前小一半 => 原来 0.02, 现在可用 0.01
    gap = 0.005
    dataset_count = len(dataset_list)
    # 每个 dataset 里 2 个柱子 (text, img), 中间无缝, dataset 间留 gap
    one_dataset_total = group_width / dataset_count
    bar_width = (one_dataset_total - gap) / 2

    def adjust_color(rgb, factor=1.0):
        r, g, b = rgb
        if factor > 1:
            # towards white
            return (r + (1 - r)*(factor-1),
                    g + (1 - g)*(factor-1),
                    b + (1 - b)*(factor-1))
        else:
            # towards black
            return (r*factor, g*factor, b*factor)

    # 定义基色
    base_colors = sns.color_palette("Set2", n_colors=dataset_count)

    text_bar_containers = []
    img_bar_containers = []

    # 为图例存储: (ds, mode) -> (BarContainer, label)
    legend_map = {}

    # ---------------------------
    # 4. 绘制并列柱状图
    # ---------------------------
    for i, model in enumerate(model_order):
        x_center = i
        group_left = x_center - group_width / 2

        for j, ds in enumerate(dataset_list):
            offset_dataset = j * (2 * bar_width + gap)
            x_text = group_left + offset_dataset
            x_img  = x_text + bar_width

            text_val = result_dict.get((model, ds, "text"), 0)
            img_val  = result_dict.get((model, ds, "img"), 0)

            # 3) Text 用深一点，Img 用浅一点
            base_col = base_colors[j]
            col_text = adjust_color(base_col, factor=0.8)  # 深
            col_img  = adjust_color(base_col, factor=1.2)  # 浅

            rect_text = ax.bar(x_text, text_val, width=bar_width,
                               color=col_text, edgecolor="none")
            rect_img  = ax.bar(x_img, img_val, width=bar_width,
                               color=col_img, edgecolor="none")

            text_bar_containers.append(rect_text)
            img_bar_containers.append(rect_img)

            legend_map[(ds, "text")] = (rect_text, f"{ds}(text)")
            legend_map[(ds, "img")]  = (rect_img,  f"{ds}(img)")

    # ---------------------------
    # 5. 坐标轴与范围
    # ---------------------------
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([model_name_map.get(m, m) for m in model_order], fontsize=9)
    ax.set_ylabel("Result", fontsize=10)
    ax.set_ylim(0, 100)  # 或根据实际需要再调大

    plt.title("All Models: multiple Datasets × 2 Modes = bars per model", fontsize=12)

    # ---------------------------
    # 6. 在柱子上方标数
    #  - 数字字体变小一些, 并且文字“横过来”（rotation=0）
    # ---------------------------
    for container in text_bar_containers:
        ax.bar_label(container, fmt="%.1f", padding=0, fontsize=7, rotation=90)

    for container in img_bar_containers:
        ax.bar_label(container, fmt="%.1f", padding=0, fontsize=7, rotation=90)

    # ---------------------------
    # 7. 图例设置
    #   - 图例变小一半, 放在左上角
    # ---------------------------
    handles = []
    labels = []
    for ds in dataset_list:
        ds_name = ds.replace("Dataset", "")  # 去掉后缀 "Dataset"
        rect_container_text, _ = legend_map[(ds, "text")]
        # 一个数据集对应2个图例条目: text在左、img在右
        handles.append(rect_container_text[0])
        labels.append(f"{ds_name}(text)")

        rect_container_img, _ = legend_map[(ds, "img")]
        handles.append(rect_container_img[0])
        labels.append(f"{ds_name}(img)")

    # for ds in dataset_list:
    #     ds_name = ds.replace("Dataset", "")  # 去掉后缀 "Dataset"
    #     rect_container_img, _ = legend_map[(ds, "img")]
    #     handles.append(rect_container_img[0])
    #     labels.append(f"{ds_name}(img)")

    ax.legend(
        handles, labels,
        title="Dataset(Mode)",  # 可视需要保留或改成别的
        loc="upper left",
        bbox_to_anchor=(0, 1),
        ncol=3,  # 两列
        fontsize=9,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Complex barplot saved to {output_path}")

def process_data_SC31(paths=["eval_cache.json"]):
    """
    Processes the JSON data from the given paths and returns the result dictionary needed for visualization.
    """
    import pandas as pd

    # Read the data
    df = extract_df(paths)

    # ---------------------------
    # 1. Mappings and lists
    # ---------------------------
    dataset_prompt_map = {
        "SuperGLUEDataset":  "base",
        "GLUEDataset":       "base",
        "MMLUProDataset":    "cot",
        "ARCDataset":        "base",
        "MBPPDataset":       "base",
        "GSM8KDataset":      "cot",
        "TableBenchDataset": "base",  # This one is not multiplied by 100
        "WikiSS_QADataset": "base",
        "SlidesVQADataset": "base",
        "MathverseDataset": "cot",
    }

    dataset_list = [
        "SuperGLUEDataset", "GLUEDataset", "MMLUProDataset",
        "ARCDataset", "MBPPDataset", "GSM8KDataset",  # "TableBenchDataset"
    ]

    model_order = ["Qwen2_VL_2B", "Phi_3_5_vision", "Qwen2_VL_7B", "Gemini_Flash", "GPT4o"]

    # ---------------------------
    # 2. Filter data
    # ---------------------------
    df["prompt_ok"] = df["dataset"].map(dataset_prompt_map) == df["prompt"]
    df = df[df["dataset"].isin(dataset_list) & df["prompt_ok"]]
    df = df[df["model"].isin(model_order)]

    # Multiply result by 100 except for TableBenchDataset
    mask_not_table = df["dataset"] != "TableBenchDataset"
    df.loc[mask_not_table, "result"] = df.loc[mask_not_table, "result"] * 100

    # Aggregate results
    df_agg = df.groupby(["model", "dataset", "mode"], as_index=False)["result"].mean()

    # Create result_dict for visualization
    result_dict = {}
    for row in df_agg.itertuples():
        result_dict[(row.model, row.dataset, row.mode)] = row.result

    return result_dict

def process_data_SC32(paths=["eval_cache.json"]):
    """
    Processes the JSON data from the given paths and returns the result dictionary needed for visualization.
    Specifically, it finds all pairs of (all given models, 'TableBenchDataset' - {all subsets except 'final_score'}, 'text'/'semi'/'img') and corresponding results.
    """
    import pandas as pd

    # Read the data
    df = extract_df(paths)
    # print(df[df['prompt']=='semi'][:10])
    # exit(0)

    # ---------------------------
    # 1. Mappings and lists
    # ---------------------------
    # Focus on the TableBenchDataset
    dataset_list = ['TableBenchDataset']

    # Define the models to include
    model_order = ["Qwen2_VL_2B", "Phi_3_5_vision", "Qwen2_VL_7B", "Gemini_Flash", "GPT4o"]

    # Define the modes we're interested in
    mode_list = ['text', 'semi', 'img']

    # ---------------------------
    # 2. Filter data
    # ---------------------------
    # Filter for the desired dataset, models, and modes
    df = df[df["dataset"].isin(dataset_list)]
    df = df[df["model"].isin(model_order)]
    df = df[df["mode"].isin(mode_list)]

    # Exclude the 'final_score' subset
    df = df[df['subset_name'] != 'final_score']

    # For TableBenchDataset, we don't need to multiply results by 100
    # So we can proceed directly to aggregating the results

    # ---------------------------
    # 3. Aggregate results
    # ---------------------------
    # Group by model, dataset, mode, and subset_name to get the mean result
    df_agg = df.groupby(
        ["model", "dataset", "mode", "subset_name"], as_index=False
    )["result"].mean()

    # ---------------------------
    # 4. Create result_dict for visualization
    # ---------------------------
    # We'll map (model, subset_name, mode) to result since we're focusing on TableBenchDataset
    result_dict = {}
    for row in df_agg.itertuples():
        result_dict[(row.model, row.subset_name, row.mode)] = row.result

    return result_dict

def process_data_SC33(paths=["eval_cache.json"]):
    """
    Processes the JSON data from the given paths and returns the result dictionary needed for visualization.
    """
    import pandas as pd

    # Read the data
    df = extract_df(paths)

    # ---------------------------
    # 1. Mappings and lists
    # ---------------------------
    dataset_prompt_map = {
        "SuperGLUEDataset":  "base",
        "GLUEDataset":       "base",
        "MMLUProDataset":    "cot",
        "ARCDataset":        "base",
        "MBPPDataset":       "base",
        "GSM8KDataset":      "cot",
        "TableBenchDataset": "base",  # This one is not multiplied by 100
        "WikiSS_QADataset": "base",
        "SlidesVQADataset": "base",
        "MathverseDataset": "cot",
    }

    dataset_list = [
        "MathverseDataset", "SlidesVQADataset", "WikiSS_QADataset",
    ]

    model_order = ["Qwen2_VL_2B", "Phi_3_5_vision", "Qwen2_VL_7B", "Gemini_Flash", "GPT4o"]

    # ---------------------------
    # 2. Filter data
    # ---------------------------
    df["prompt_ok"] = df["dataset"].map(dataset_prompt_map) == df["prompt"]
    df = df[df["dataset"].isin(dataset_list) & df["prompt_ok"]]
    df = df[df["model"].isin(model_order)]

    # Multiply result by 100 except for TableBenchDataset
    mask_not_table = df["dataset"] == "MathverseDataset"
    df.loc[mask_not_table, "result"] = df.loc[mask_not_table, "result"] * 100

    # Aggregate results
    df_agg = df.groupby(["model", "dataset", "mode"], as_index=False)["result"].mean()

    # Create result_dict for visualization
    result_dict = {}
    for row in df_agg.itertuples():
        result_dict[(row.model, row.dataset, row.mode)] = row.result

    return result_dict

def plot_figure(result_dict, output_path="plt.png", model_order=["Qwen2_VL_2B", "Phi_3_5_vision", "Qwen2_VL_7B",
                                                                 "Gemini_Flash", "GPT4o"], dataset_order=None, model_name_map=None, ncol=2):
    """
    Generates a grouped bar plot figure based on the result_dict and saves it to the specified output path.
    Adapts spacing to ensure readability.

    Parameters:
        result_dict (dict): A dictionary with keys as (model, dataset, mode) and values as results.
        output_path (str): File path to save the plot.
        model_order (list): Ordered list of models to display.
        dataset_order (list): Ordered list of datasets to display.
        model_name_map (dict): Mapping of model names for display.
        ncol (int): Number of columns in the legend.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract unique models, datasets, and modes
    models = sorted(set([model for model, _, _ in result_dict.keys()]))
    datasets = sorted(set([dataset for _, dataset, _ in result_dict.keys()]))
    modes = sorted(set([mode for _, _, mode in result_dict.keys()]))

    if model_order:
        models = model_order
    if dataset_order:
        datasets = dataset_order

    # Order modes
    if set(modes) == {'text', 'semi', 'img'}:
        modes = ['text', 'semi', 'img']
    elif set(modes) == {'text', 'img'}:
        modes = ['text', 'img']

    if model_name_map is None:
        model_name_map = {model: model for model in models}

    # Helper function for color adjustment
    def adjust_color(rgb, factor=1.0):
        r, g, b = rgb
        if factor > 1:
            return (min(r + (1 - r) * (factor - 1), 1),
                    min(g + (1 - g) * (factor - 1), 1),
                    min(b + (1 - b) * (factor - 1), 1))
        else:
            return (r * factor, g * factor, b * factor)

    sns.set_style("whitegrid")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)

    group_width = 0.8  # Total width allocated to each group (model)
    inter_model_gap = 0.01  # Space between models
    inter_dataset_gap = 0.005  # Space between datasets within a model
    num_datasets = len(datasets)
    num_modes = len(modes)

    per_dataset_width = (group_width - (num_datasets - 1) * inter_dataset_gap) / num_datasets
    per_mode_width = per_dataset_width / num_modes

    base_colors = sns.color_palette("Set2", n_colors=num_datasets)

    legend_map = {}

    # Plot bars
    for i, model in enumerate(models):
        group_left = i - group_width / 2

        for j, dataset in enumerate(datasets):
            dataset_left = group_left + j * (per_dataset_width + inter_dataset_gap)
            base_color = base_colors[j]

            for k, mode in enumerate(modes):
                x = dataset_left + k * per_mode_width
                value = result_dict.get((model, dataset, mode), 0)

                color = adjust_color(base_color, 0.8 + 0.4 * k / (num_modes - 1))

                rect = ax.bar(x, value, width=per_mode_width, color=color, edgecolor="none")

                # Add legend mapping
                legend_key = (dataset, mode)
                if legend_key not in legend_map:
                    legend_label = f"{dataset.replace('Dataset', '')}({mode})"
                    legend_map[legend_key] = (rect, legend_label)

    # Set axis labels
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_name_map.get(m, m) for m in models], fontsize=14)
    ax.set_ylabel("Result", fontsize=14)
    ax.set_ylim(0, 100)

    # Annotate bars
    for rect in ax.patches:
        height = rect.get_height()
        # if height > 0:
        ax.text(rect.get_x() + rect.get_width() / 2, height,
                f"{height:.2f}", ha="center", va="bottom", fontsize=11, rotation=90)

    # Set up legend
    handles, labels = zip(*[(container[0], label) for container, label in legend_map.values()])
    # ax.legend(handles, labels, title="Dataset(Mode)", loc="upper left",
    #           bbox_to_anchor=(0, 1.2), ncol=ncol, fontsize=11, title_fontsize=13)
    ax.legend(handles, labels, title="Dataset(Mode)", loc="upper left",
              bbox_to_anchor=(0, 1), ncol=ncol, fontsize=11, title_fontsize=13)
    # ax.spines['top'].set_visible(False)

    # plt.title("Models Comparison across Datasets and Modes on Text-Only Input", fontsize=16, pad=60)
    plt.title("Models Comparison across Datasets and Modes on TableBench", fontsize=16)
    plt.title("Models Comparison across Datasets and Modes on Multimodal Input", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # extract_df()
    # vis_evalcache(paths = ["eval_cache.json", "eval_cache_vector.json"], control="prompt", level="mode",
    #                output_path="plt1.png")
    # #
    # vis_evalcache(paths = ["eval_cache.json", "eval_cache_vector.json"], control="mode", level="prompt",
    #               output_path="plt2.png")
    # find_badcase()

    # vis_barplot(paths=["eval_cache.json", "eval_cache_vector.json"], # control="mode", level="prompt",
    #               output_path="plt_text.png", mode="text")
    # vis_barplot(paths=["eval_cache.json", "eval_cache_vector.json"], # control="mode", level="prompt",
    #               output_path="plt_img.png", mode="img")

    # vis_double_barplot(paths=["eval_cache.json", "eval_cache_vector.json"], output_path="plt_double.png")
    # vis_single_barplot(paths=["eval_cache.json", "eval_cache_vector.json"], output_path="plt_single_6.png")

    # result_dict = process_data_SC31(paths=["eval_cache.json", "eval_cache_vector.json"])
    # print(result_dict)
    # plot_figure(result_dict, output_path = "plt_decompose.png", dataset_order=["GLUEDataset", "SuperGLUEDataset",
    #                                                                             "ARCDataset", "GSM8KDataset",
    #                                                                            "MBPPDataset", "MMLUProDataset"])

    # result_dict = process_data_SC32(paths=["eval_cache.json", "eval_cache_vector.json"])
    # print(result_dict)
    # plot_figure(result_dict, output_path = "plt_decompose_sc32.png", ncol=4, dataset_order=[
    #     "FactChecking", "DataAnalysis", "NumericalReasoning", "Visualization"])

    result_dict = process_data_SC33(paths=["eval_cache.json", "eval_cache_vector.json"])
    print(result_dict)
    result_dict[ ('Qwen2_VL_2B', 'MathverseDataset', 'text')] = 51.65
    plot_figure(result_dict, output_path = "plt_decompose_sc33.png", ncol=6, dataset_order=[
        "MathverseDataset", "SlidesVQADataset", "WikiSS_QADataset"])

