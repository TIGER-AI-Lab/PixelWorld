import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from io import StringIO
import numpy as np


def save_table_to_csv(table_json, csv_name='table.csv'):
    """
    将 table_json 中的 columns、data 写入 CSV 文件。
    示例的 table_json 结构:
    {
      "columns": ["Month", "Jan", ...],
      "data": [
         ["Record high °F (°C)", "76\n(24)", "86\n(30)", ...],
         ...
      ]
    }
    """
    columns = table_json["columns"]
    data = table_json["data"]
    rows = []
    for row in data:
        rows.append(dict(zip(columns, row)))
    df = pd.DataFrame(rows)
    df.to_csv(csv_name, index=False)


def extract_python_code(prediction_str):
    """
    从模型输出中提取三重反引号里的 python 代码。
    """
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, prediction_str, flags=re.DOTALL)
    return match.group(1) if match else ""


def get_bar_y_predictions():
    """
    从当前 matplotlib 画布中提取所有 bar 的高度。
    """
    patches = plt.gca().patches
    return [patch.get_height() for patch in patches]


def compare_arrays(ref, pred, rtol=1e-2):
    """
    对比两个列表中对应元素是否相等（可加一定容忍度 rtol）。
    """
    if len(ref) != len(pred):
        return False
    for r_val, p_val in zip(ref, pred):
        if abs(r_val - p_val) > rtol:  # 允许一定浮点误差
            return False
    return True


def evaluate_visualization(table_json, y_references, prediction_str):
    """
    评测可视化代码:
    1) 先把表格写到 table.csv
    2) 提取并执行模型生成的可视化代码 --> ecr@1(能否成功执行)
    3) 在图像上提取 bar 的 y 值与参考对比 --> pass@1(画的对不对)

    返回:
    {
      "ecr@1": bool,
      "pass@1": bool
    }
    """
    if type(table_json) == str:
        table_json = json.loads(table_json)
    # print("y_references:", y_references)
    if type(y_references) == str:
        y_references = eval(y_references.split('=')[-1])

    # ------ 1) 保存表格 ------
    save_table_to_csv(table_json, 'table.csv')

    # ------ 2) 提取 python 代码 ------
    code_snippet = extract_python_code(prediction_str)
    if not code_snippet.strip():
        # 如果没找到有效代码，则 ecr@1=False, pass@1=False
        return {"ecr@1": False, "pass@1": False}

    # ------ 3) 执行代码 ------
    backup_stdout = sys.stdout
    sys.stdout = StringIO()  # 重定向输出，防止乱打日志
    ecr_1 = True  # 先默认能执行
    try:
        exec(code_snippet, globals())
    except Exception as e:
        ecr_1 = False
    finally:
        sys.stdout = backup_stdout

    # 如果执行报错，则 ecr_1=False, pass@1 也必定=False
    if not ecr_1:
        plt.close('all')
        return {"ecr@1": False, "pass@1": False}

    # ------ 4) 如果执行成功，提取图中数据，对比参考 ------
    #     这里仅示例 bar chart 场景
    pass_1 = False
    try:
        predicted_y = get_bar_y_predictions()
        plt.close('all')  # 清理画布
        # 假设 y_references 是二维 list，如 [[24,30,36,...]]
        if y_references and len(y_references) > 0:
            ref_array = y_references[0]
            pass_1 = compare_arrays(ref_array, predicted_y)
    except Exception as e:
        pass_1 = False  # 如果这里再报错就不匹配

    return {"ecr@1": ecr_1, "pass@1": pass_1}

if __name__ == "__main__":
    # =============== 1) 修改后的 Table JSON ===============
    # 只包含两列: Month、Record high °F (°C)
    # 每一行代表1个月，第二列里是形如 "(24°C)" 的字符串，方便用正则解析
    table_example = {
        "columns": ["Month", "Record high °F (°C)"],
        "data": [
            ["Jan", "(24°C)"],
            ["Feb", "(30°C)"],
            ["Mar", "(36°C)"],
            ["Apr", "(36°C)"],
            ["May", "(35°C)"],
            ["Jun", "(40°C)"],
            ["Jul", "(44°C)"],
            ["Aug", "(43°C)"],
            ["Sep", "(41°C)"],
            ["Oct", "(36°C)"],
            ["Nov", "(32°C)"],
            ["Dec", "(26°C)"],
        ]
    }

    # =============== 2) 参考值 y_references ===============
    # 对应 12个月的摄氏度
    y_refs = [[24, 30, 36, 36, 35, 40, 44, 43, 41, 36, 32, 26]]

    # =============== 3) 模型输出(代码段) ===============
    # 这里的代码会从 table.csv 读入两列 "Month", "Record high °F (°C)"
    # 再用正则解析出 (xx°C) 中的 xx 作为数值
    predicted_text = """```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('table.csv')

months = df['Month']
record_high_celsius = df['Record high °F (°C)'].str.extract(r'\\((.*?)\\)')[0].str.replace('°C','').astype(float)

plt.bar(months, record_high_celsius)
plt.show()
```"""

    # =============== 4) 调用 evaluate_visualization ===============
    # 注意：需要你上面定义的函数 e.g. evaluate_visualization(table_json, y_references, prediction_str)
    # 这里假设脚本的名字和函数一并存在
    result = evaluate_visualization(
        table_json=table_example,
        y_references=y_refs,
        prediction_str=predicted_text
    )

    print("Evaluation result:", result)
    # 预计输出:
    # Evaluation result: {'ecr@1': True, 'pass@1': True}
