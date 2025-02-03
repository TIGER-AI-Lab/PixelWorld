import os
import json
from typing import Union, List, Optional

def summarize_token_usage(folder_path):
    """
    Summarizes the token usage from all token_usage.jsonl files in the given folder.

    Args:
        folder_path (str): Path to the folder containing the token_usage.jsonl files.

    Returns:
        dict: A dictionary where the keys are filenames and the values are the total prompt and completion tokens.
    """
    summary = {}
    overall_prompt_tokens = 0
    overall_completion_tokens = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('_token_usage.jsonl'):
            file_path = os.path.join(folder_path, filename)
            total_prompt_tokens = 0
            total_completion_tokens = 0

            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        if 'key' in data and data['key'] == 'usage':
                            usage = data.get('value', {})
                            total_prompt_tokens += usage.get('prompt_tokens', 0)
                            total_completion_tokens += usage.get('completion_tokens', 0)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")

            summary[filename] = {
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens
            }
            overall_prompt_tokens += total_prompt_tokens
            overall_completion_tokens += total_completion_tokens

    summary['Overall'] = {
        'total_prompt_tokens': overall_prompt_tokens,
        'total_completion_tokens': overall_completion_tokens
    }

    return summary

class GeminiKeyScheduler:
    def __init__(self):
        """
        初始化时加载免费 Key 和付费 Key。
        免费 Key 使用轮询策略，付费 Key 有单独的计数限制。
        """
        # 从环境变量加载免费和付费 Key
        free_keys_str = os.getenv("GEMINI_API_KEYS_FREE", "[]")
        try:
            self.free_keys = json.loads(free_keys_str)
        except:
            self.free_keys = []  # 如果解析失败，免费 Key 列表为空

        self.paid_key = os.getenv("GEMINI_API_KEY_PRICED", None)

        print(f"Loaded {len(self.free_keys)} free keys and 1 paid key: {self.paid_key is not None}")

        if not self.free_keys and not self.paid_key:
            raise ValueError("No available Gemini keys (both free and paid are empty)!")

        # === 新增计数 & 状态 ===
        # 1. 免费 Key 相关
        self.current_free_key_index = 0  # 当前免费 Key 的轮询索引
        self.free_keys_usage = {key: 0 for key in self.free_keys}

        # 2. 付费 Key 相关
        self.paid_key_usage = 0       # 付费 Key 成功使用次数，可根据需要使用
        self.paid_key_limit = 100     # 默认付费 Key 每轮次最多用 100 次，可自定义
        self.paid_key_fail_count = 0  # 付费 Key 连续失败次数

        # 3. 最近一次分配信息
        self.last_used_key = None
        self.is_paid_key = False  # 标识是否刚用了付费 Key

        # 4. 轮次相关：如果连续 10 轮都返回 None，则抛错
        self.none_round_count = 0  # 累计“返回 None”的轮次数
        self.max_none_round = 10   # 连续多少轮返回 None 之后抛错误

    def ask_for_key(self) -> Optional[str]:
        """
        返回一个可以使用的 Key。
        1. 优先尝试免费 Key。
        2. 如果所有免费 Key 都不可用，则尝试付费 Key。
        3. 如果付费 Key 连续失败 3 次，则本次返回 None，下次重新从免费 Key 开始。
        4. 若连续 10 轮都返回 None，则抛出 RuntimeError。
        """
        # 如果付费 Key 的使用次数已经到达限制，则重置（可根据业务需求调整）
        if self.paid_key_usage >= self.paid_key_limit:
            self.paid_key_usage = 0
            self.current_free_key_index = 0

        # 先尝试分配当前的免费 Key（如果还有没失效的）
        while self.current_free_key_index < len(self.free_keys):
            candidate_key = self.free_keys[self.current_free_key_index]
            if candidate_key:  # 简单假设只要存在就有效
                self.last_used_key = candidate_key
                self.is_paid_key = False
                return candidate_key
            else:
                self.current_free_key_index += 1

        # 如果免费 Key 都失效了，再尝试付费 Key
        if self.paid_key and (self.paid_key_fail_count < 3):
            # 只有在付费 Key 连续失败次数 < 3 时继续返回 paid_key
            self.last_used_key = self.paid_key
            self.is_paid_key = True
            return self.paid_key
        else:
            # 如果走到这里，说明：
            # 1. 付费 Key 不存在，或者
            # 2. 付费 Key 连续失败次数 >= 3

            # 如果这一次直接返回 None，需要更新 none_round_count
            self.none_round_count += 1
            # 检查是否超过最大轮次
            if self.none_round_count >= self.max_none_round:
                raise RuntimeError("No available keys for 10 consecutive rounds. Stop trying.")

            # 返回 None 之后，下次请求重新从第一个免费 Key 开始
            self.reset_after_none()
            return None

    def feedback(self, success: bool):
        """
        根据推理结果的成功与否，调整 Key 的使用状态。
        """
        if self.last_used_key is None:
            # 理论上不会出现，但为了安全做个防御
            raise RuntimeError("No key was assigned before feedback.")

        if success:
            # === 如果推理成功 ===
            # 清空“返回 None”的轮次数
            self.none_round_count = 0

            if self.is_paid_key:
                # 付费 Key 成功，更新状态
                self.paid_key_usage += 1
                self.paid_key_fail_count = 0  # 连续失败清零
            else:
                # 免费 Key 成功
                self.free_keys_usage[self.last_used_key] += 1
            return

        # === 如果推理失败 ===
        if not self.is_paid_key:
            # 免费 Key 失效：切换到下一个免费 Key
            self.current_free_key_index += 1
            import time
            time.sleep(10)  # 避免太快失败，可以根据需要调整
        else:
            # 付费 Key 失败
            self.paid_key_fail_count += 1
            if self.paid_key_fail_count >= 3:
                # 一旦付费 Key 连续失败达到 3 次，则弃用付费 Key
                # 并且本轮后续 ask_for_key() 会返回 None（直到下一轮重新开始）
                pass

    def reset_after_none(self):
        """
        每次返回 None 之后，重置一些状态，以便下一次询问重新开始。
        """
        # 清空付费 Key 相关的使用计数与失败计数
        # 但注意，这里并没有把 self.paid_key = None，而是保留它
        # 下次问的时候，如果 free_key 都失效，还是会继续尝试付费 key（其 fail_count 已清 0）
        self.paid_key_fail_count = 0
        self.paid_key_usage = 0

        # 重新回到免费 Key 第一个位置
        self.current_free_key_index = 0

        # 也可以视需求决定是否要“彻底弃用付费 Key”
        # self.paid_key = None
        # 这里仅仅清空计数，让下次可以重新尝试

        # 上一次用的 key 就清空
        self.last_used_key = None
        self.is_paid_key = False

# Example usage
if __name__ == "__main__":
    # folder_path = "cache_gpt4o"
    # usage_summary = summarize_token_usage(folder_path)
    # for filename, stats in usage_summary.items():
    #     print(f"{filename}: Prompt Tokens = {stats['total_prompt_tokens']}, Completion Tokens = {stats['total_completion_tokens']}")
    test_gemini_key_scheduler()
