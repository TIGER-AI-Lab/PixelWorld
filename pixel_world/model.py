import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
import google.generativeai as genai
from openai import AzureOpenAI
from qwen_vl_utils import process_vision_info
import PIL.Image as Image
import os
import torch
from typing import Union, List, Optional
from PIL import Image
from io import BytesIO
import base64
import time
import openai
import math
import base64
import hashlib
from typing import List
import torch
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)

import importlib

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import infer_auto_device_map, dispatch_model

class ModelWrapperBase(ABC):
    def __init__(self):
        super().__init__()

    def _batch_process(self, items):
        """
        将items按照self.batch_size进行分批处理的一个简单工具函数
        """
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    @abstractmethod
    def VisionInference(self, queries, image_paths, image_generator = None):
        # List of text prompt, image prompt path and image generator to fix the image and subsample
        # return list of answer
        pass

    @abstractmethod
    def VisionMultiInference(self, queries, image_paths, image_generator = None): # TODO
        # Difference is that here image_paths is a list of list of image paths, means input can be multiple images
        # return list of answer
        pass

    @abstractmethod
    def TextInference(self, queries):  # return a list of answer
        pass

class QWen2VLWrapper(ModelWrapperBase):
    # name = "Qwen2-VL-7B-Instruct"
    def __init__(self, model_name = "Qwen2-VL-7B-Instruct", model_path="Qwen/Qwen2-VL-7B-Instruct", batch_size=8):
        super().__init__()
        self.name = model_name
        print("Initializing Qwen2-VL model...")
        self.batch_size = batch_size
        self.max_pixels = 1254 * 1254
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        # self.model.to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Model initialized successfully.")

    def _batch_image_process(self, batch_queries, batch_image_paths):
        prepared_messages_list = []
        for q, img_path in zip(batch_queries, batch_image_paths):
            prepared_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": q}
                    ]
                }
            ]
            prepared_messages_list.append(prepared_messages)

        text_list = []
        image_inputs_list = []
        video_inputs_list = []

        for pm in prepared_messages_list:
            txt = self.processor.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(pm)
            text_list.append(txt)
            image_inputs_list.append(image_inputs[0] if image_inputs else None)
            video_inputs_list.append(video_inputs[0] if video_inputs else None)

        if all(v is None for v in video_inputs_list):
            video_inputs_list = None

        inputs = self.processor(
            text=text_list,
            images=image_inputs_list,
            videos=video_inputs_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        return inputs

    def VisionInference(self, queries, image_paths, image_generator = None):
        if len(queries) != len(image_paths):
            raise ValueError("The length of queries and image_paths must match.")

        # print("Running vision inference for multiple queries with batch size:", self.batch_size)
        all_results = []

        # 同时对queries和image_paths进行分批处理
        query_batches = list(self._batch_process(queries))
        image_batches = list(self._batch_process(image_paths))

        for batch_queries, batch_image_paths in zip(query_batches, image_batches):
            # batch_image_paths_subsampled = [image_generator.subsample_image(img_path, self.max_pixels, "qwen2-vl"
            #                                     ) if image_generator else img_path for img_path in batch_image_paths]
            inputs = self._batch_image_process(batch_queries, batch_image_paths)
            # print(len(inputs["input_ids"][0]))
            try:
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,# temperature=0.0
                )
            # except torch.cuda.OutOfMemoryError as oom_e:
            #     subsampled_image_paths = [image_generator.subsample_image(img_path, self.max_pixels, "qwen2-vl")
            #                                 if image_generator else img_path for img_path in batch_image_paths]
            #     print("OOM Error, trying to subsample images and retrying inference...")
            #     print("Subsampled image paths:", subsampled_image_paths)
            #     inputs = self._batch_image_process(batch_queries, subsampled_image_paths)
            #     # print(len(inputs["input_ids"][0]))
            #     generated_ids = self.model.generate(
            #         **inputs, max_new_tokens=1024, do_sample=False,# temperature=0.0
            #     )
            except Exception as e:
                print("Error during inference:", e)
                print(inputs)
                exit(0)

            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            batch_results = []
            for o in output_texts:
                # 假设输出格式与原代码相同
                result = o.split("assistant\n")[-1]
                batch_results.append(result)

            all_results.extend(batch_results)

        return all_results

    def VisionMultiInference(self, queries, image_paths, image_generator=None):
        def _batch_image_process(batch_queries, batch_image_paths):
            prepared_messages_list = []
            for q, img_paths in zip(batch_queries, batch_image_paths):
                # img_paths is a list of image paths
                prepared_messages = [
                    {
                        "role": "user",
                        "content": [
                            # include multiple images
                            *[{"type": "image", "image": img_path} for img_path in img_paths],
                            {"type": "text", "text": q}
                        ]
                    }
                ]
                prepared_messages_list.append(prepared_messages)

            text_list = []
            image_inputs_list = []
            video_inputs_list = []

            for pm in prepared_messages_list:
                txt = self.processor.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(pm)
                text_list.append(txt)
                image_inputs_list.append(image_inputs if image_inputs else None)
                video_inputs_list.append(video_inputs if video_inputs else None)

            if all(v is None for v in video_inputs_list):
                video_inputs_list = None

            inputs = self.processor(
                text=text_list,
                images=image_inputs_list,
                videos=video_inputs_list,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            return inputs

        """
        The difference is that here image_paths is a list of lists of image paths, meaning input can be multiple images per query.
        Returns a list of answers.
        """
        if len(queries) != len(image_paths):
            raise ValueError("The length of queries and image_paths must match.")

        all_results = []

        # Batch processing
        query_batches = list(self._batch_process(queries))
        image_batches = list(self._batch_process(image_paths))

        for batch_queries, batch_image_paths in zip(query_batches, image_batches):
            # batch_image_paths is a list where each element is a list of image paths
            inputs = _batch_image_process(batch_queries, batch_image_paths)
            try:
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,  # temperature=0.0
                )
            except Exception as e:
                print("Error during inference:", e)
                print(inputs)
                exit(0)

            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            batch_results = []
            for o in output_texts:
                # Process the output to extract the assistant's reply
                result = o.split("assistant\n")[-1]
                batch_results.append(result)

            all_results.extend(batch_results)

        return all_results

    def TextInference(self, queries):
        # print("Running text inference for multiple queries with batch size:", self.batch_size)
        all_results = []

        for batch_queries in self._batch_process(queries):
            prepared_messages_list = []
            for q in batch_queries:
                prepared_messages = [{"role": "user", "content": q}]
                prepared_messages_list.append(prepared_messages)

            text_list = []
            for pm in prepared_messages_list:
                txt = self.processor.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
                text_list.append(txt)

            inputs = self.processor(
                text=text_list,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=4096, do_sample=False, # temperature=0.0
            )
            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            batch_results = []
            for o in output_texts:
                result = o.split("assistant\n")[-1]
                batch_results.append(result)

            all_results.extend(batch_results)

        return all_results

class GPT4oWrapper(ModelWrapperBase):
    name = "gpt-4o"

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        num_samples: int = 1,
        api_version: str = "2024-05-01-preview",
        text_cache_path: str = "text_cache.jsonl",
        img_cache_path: str = "img_cache.jsonl",
        token_usage_path: str = "token_used.jsonl",
        max_retries: int = 10,
        sync_interval: int = 100
    ):
        """
        sync_interval: 每多少次查询后，与文件做一次重读合并
        """
        super().__init__()
        # endpoint = os.getenv("ENDPOINT_URL", "https://azure-gpt4.openai.azure.com/")
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        # subscription_key = os.environ["AZURE_KEY"]
        #
        # self.client = AzureOpenAI(
        #     azure_endpoint=endpoint,
        #     api_key=subscription_key,
        #     api_version=api_version,
        # )
        from openai import OpenAI
        api_key = os.getenv("OPENAI_KEY")
        self.client = OpenAI(api_key=api_key)


        self.model_name = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.max_retries = max_retries
        self.sync_interval = sync_interval

        # 缓存文件路径
        self.text_cache_path = text_cache_path
        self.img_cache_path = img_cache_path
        self.token_usage_path = token_usage_path

        # -----------------------
        # 1) 启动时先载入到内存
        # -----------------------
        self.text_cache = self._load_jsonl(text_cache_path)  # key -> value
        self.img_cache = self._load_jsonl(img_cache_path)    # key -> value
        # token usage 不一定需要常驻内存，可仅做写；若要做统计，可在 init 时读一次做累计。
        self.total_token_usage = self._load_token_usage()

        # 统计查询次数，每当达到sync_interval时就重读文件并合并
        self.query_count_text = 0
        self.query_count_img = 0

    def change_cache_path(self, text_cache_path: str, img_cache_path: str, token_usage_path: str):
        self.text_cache_path = text_cache_path
        self.img_cache_path = img_cache_path
        self.token_usage_path = token_usage_path
        self.text_cache = self._load_jsonl(text_cache_path)
        self.img_cache = self._load_jsonl(img_cache_path)
        self.total_token_usage = self._load_token_usage()

    # -------------------------------------------------------------------------
    # JSONL 读/写辅助
    # -------------------------------------------------------------------------
    def _load_jsonl(self, file_path: str) -> dict:
        """
        将 jsonl 文件加载为一个 dict。
        每行都应该是形如 {"key": <key>, "value": <value>} 的 JSON。
        如果出现重复 key，后出现的会覆盖前出现的。
        """
        data_dict = {}
        if not os.path.exists(file_path):
            return data_dict

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "key" in record and "value" in record:
                        data_dict[record["key"]] = record["value"]
                except:
                    # 如果解析失败就跳过该行
                    pass
        return data_dict

    def _append_jsonl(self, file_path: str, key: str, value):
        """
        追加写入一行 JSON 到 jsonl 文件。
        每条记录统一格式：{"key": <key>, "value": <value>}
        """
        record = {"key": key, "value": value}
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _sync_cache_from_file(self, file_path: str, cache: dict):
        """
        从 file_path 重新加载到一个临时 dict，再与 cache 合并 (以文件为准或者内存为准，可自行决定策略)。
        此处示例：如果文件和内存都存在相同key，**以内存中的为准**。
        你也可以改成以文件为准。
        """
        new_data = self._load_jsonl(file_path)
        for k, v in new_data.items():
            if k not in cache:
                # 如果本地缓存没有，则添加
                cache[k] = v
            # 如果想以文件为准，改写成 `cache[k] = v`

    # -------------------------------------------------------------------------
    # Token usage 读/写
    # -------------------------------------------------------------------------
    def _load_token_usage(self) -> dict:
        """
        从 token_usage_path 中加载累计的 token usage。
        这里演示全量遍历做累加。
        """
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0}
        if not os.path.exists(self.token_usage_path):
            return usage_data

        with open(self.token_usage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("key") == "usage":
                        val = record.get("value", {})
                        usage_data["prompt_tokens"] += val.get("prompt_tokens", 0)
                        usage_data["completion_tokens"] += val.get("completion_tokens", 0)
                except:
                    pass
        return usage_data

    def _append_token_usage(self, prompt_tokens: int, completion_tokens: int, cache_key: str = None):
        """
        追加一条 token usage 到 jsonl 文件
        """
        usage_entry = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        record = {
            "key": "usage",
            "value": usage_entry,
            "cache_key": cache_key if cache_key else ""
        }
        with open(self.token_usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_usage(self, prompt_tokens: int, completion_tokens: int, cache_key: str = None):
        """
        记录本次 usage 并更新内存中的累计值
        """
        # 写到文件
        self._append_token_usage(prompt_tokens, completion_tokens, cache_key)
        # 更新内存计数
        self.total_token_usage["prompt_tokens"] += prompt_tokens
        self.total_token_usage["completion_tokens"] += completion_tokens

    # -------------------------------------------------------------------------
    # GPT API
    # -------------------------------------------------------------------------

    def _call_gpt_api(self, messages):
        """
        调用 Azure OpenAI Chat Completion 接口的通用方法，带重试机制。
        如果触发了 ResponsibleAIPolicyViolation，则返回字符串 "ResponsibleAIPolicyViolation"。
        """
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=False
                )
                return completion

            except openai.BadRequestError as e:
                # 判断错误信息中是否包含 ResponsibleAIPolicyViolation
                if "ResponsibleAIPolicyViolation" in str(e):
                    return "ResponsibleAIPolicyViolation"
                else:
                    # 如果不是触发这个错误，就执行正常的重试或抛出异常逻辑
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e

            except Exception as e:
                # 处理其他类型的错误，同样进行重试
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise e  # 重试多次后仍失败，抛出异常

    # -------------------------------------------------------------------------
    # Text Inference
    # -------------------------------------------------------------------------
    def TextInference(self, queries: List[str]) -> List[str]:
        all_results = []
        for user_prompt in tqdm(queries):
            # 1) 查询前判断是否需要重读合并文件
            if self.query_count_text > 0 and self.query_count_text % self.sync_interval == 0:
                # 重新从文件同步到内存
                self._sync_cache_from_file(self.text_cache_path, self.text_cache)

            self.query_count_text += 1

            # 2) 查看内存缓存
            if user_prompt in self.text_cache:
                all_results.append(self.text_cache[user_prompt])
                continue

            # 3) 调API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
            completion = self._call_gpt_api(messages)
            if type(completion) == str and completion == "ResponsibleAIPolicyViolation" or completion is None:
                all_results.append("ResponsibleAIPolicyViolation")
                self.text_cache[user_prompt] = "ResponsibleAIPolicyViolation"
                self._append_jsonl(self.text_cache_path, user_prompt, "ResponsibleAIPolicyViolation")
                continue
            elif completion.choices and len(completion.choices) > 0:
                # print(completion)
                try:
                    answer = completion.choices[0].message.content.strip()
                except:
                    all_results.append("ResponsibleAIPolicyViolation")
                    self.text_cache[user_prompt] = "ResponsibleAIPolicyViolation"
                    self._append_jsonl(self.text_cache_path, user_prompt, "ResponsibleAIPolicyViolation")
                    continue
            else:
                answer = ""

            # 4) 记录token usage
            if hasattr(completion, "usage") and completion.usage:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            else:
                prompt_tokens = len(user_prompt.split())
                completion_tokens = len(answer.split())
            self._record_usage(prompt_tokens, completion_tokens, user_prompt)

            # 5) 写入内存缓存 + 追加写到文件
            self.text_cache[user_prompt] = answer
            self._append_jsonl(self.text_cache_path, user_prompt, answer)

            all_results.append(answer)

        return all_results

    # -------------------------------------------------------------------------
    # Vision Inference
    # -------------------------------------------------------------------------
    def VisionInference(self, queries: List[str], image_paths: List[str], image_generator = None) -> List[str]:
        if len(queries) != len(image_paths):
            raise ValueError("Length of queries and image_paths must match for VisionInference.")

        all_results = []
        for user_prompt, img_path in tqdm(zip(queries, image_paths)):
            # 1) 查询前判断是否需要重读合并文件
            if self.query_count_img > 0 and self.query_count_img % self.sync_interval == 0:
                # 重新从文件同步到内存
                self._sync_cache_from_file(self.img_cache_path, self.img_cache)

            self.query_count_img += 1

            # 2) 检查内存缓存
            cache_key = f"{img_path}##{user_prompt}"
            if cache_key in self.img_cache:
                all_results.append(self.img_cache[cache_key])
                continue

            # 3) 读图并 base64
            with open(img_path, 'rb') as f:
                img_data = f.read()
            encoded_image = base64.b64encode(img_data).decode('ascii')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
            completion = self._call_gpt_api(messages)
            if type(completion) == str and completion == "ResponsibleAIPolicyViolation":
                all_results.append("ResponsibleAIPolicyViolation")
                self.img_cache[cache_key] = "ResponsibleAIPolicyViolation"
                self._append_jsonl(self.img_cache_path, cache_key, "ResponsibleAIPolicyViolation")
                continue
            # print(completion)
            if completion.choices and len(completion.choices) > 0:
                answer = completion.choices[0].message.content.strip()
            else:
                answer = ""

            # 4) 记录tokens
            if hasattr(completion, "usage") and completion.usage:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            else:
                prompt_tokens = len(user_prompt.split()) + 1445
                completion_tokens = len(answer.split())
            self._record_usage(prompt_tokens, completion_tokens, cache_key)

            # 5) 更新内存 + 写入文件
            self.img_cache[cache_key] = answer
            self._append_jsonl(self.img_cache_path, cache_key, answer)

            all_results.append(answer)

        return all_results

    def VisionMultiInference(self, queries: List[str], image_paths: List[List[str]], image_generator=None) -> List[str]:
        """
        Performs inference on multiple queries, each potentially associated with multiple images.

        Args:
            queries (List[str]): A list of user prompts or queries.
            image_paths (List[List[str]]): A list where each element is a list of image paths corresponding to the query.
            image_generator: Optional image generator or processor.

        Returns:
            List[str]: A list of answers generated by the model for each query.
        """
        if len(queries) != len(image_paths):
            raise ValueError("Length of queries and image_paths must match for VisionMultiInference.")

        all_results = []
        for user_prompt, img_paths in tqdm(zip(queries, image_paths), total=len(queries)):
            # 1) Periodically sync cache from the file
            if self.query_count_img > 0 and self.query_count_img % self.sync_interval == 0:
                self._sync_cache_from_file(self.img_cache_path, self.img_cache)

            self.query_count_img += 1

            # 2) Check in-memory cache
            # Create a unique cache key using a hash of the sorted image paths and the user prompt
            sorted_img_paths = sorted(img_paths)
            joined_paths = ''.join(sorted_img_paths)
            images_hash = hashlib.md5(joined_paths.encode()).hexdigest()
            cache_key = f"{images_hash}##{user_prompt}"

            if cache_key in self.img_cache:
                all_results.append(self.img_cache[cache_key])
                continue

            # 3) Read and encode images
            content_entries = []
            for img_path in img_paths:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                encoded_image = base64.b64encode(img_data).decode('ascii')
                content_entries.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })

            # Append the user prompt as text
            content_entries.append({
                "type": "text",
                "text": user_prompt
            })

            messages = [
                {
                    "role": "user",
                    "content": content_entries
                }
            ]

            # 4) Call the GPT API
            completion = self._call_gpt_api(messages)
            if isinstance(completion, str) and completion == "ResponsibleAIPolicyViolation":
                all_results.append("ResponsibleAIPolicyViolation")
                self.img_cache[cache_key] = "ResponsibleAIPolicyViolation"
                self._append_jsonl(self.img_cache_path, cache_key, "ResponsibleAIPolicyViolation")
                continue

            if completion.choices and len(completion.choices) > 0:
                answer = completion.choices[0].message.content.strip()
            else:
                answer = ""

            # 5) Record token usage
            if hasattr(completion, "usage") and completion.usage:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            else:
                # Estimate tokens if usage data is unavailable
                prompt_tokens = len(user_prompt.split()) + 1445  # Adjust this estimation as needed
                completion_tokens = len(answer.split())
            self._record_usage(prompt_tokens, completion_tokens, cache_key)

            # 6) Update cache and write to file
            self.img_cache[cache_key] = answer
            self._append_jsonl(self.img_cache_path, cache_key, answer)

            all_results.append(answer)

        return all_results

class GeminiWrapper(GPT4oWrapper):
    def __init__(
        self,
        name = "gemini-1.5-flash-002",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        num_samples: int = 1,
        text_cache_path: str = "text_cache_gemini.jsonl",
        img_cache_path: str = "img_cache_gemini.jsonl",
        token_usage_path: str = "token_used_gemini.jsonl",
        max_retries: int = 10,
        sync_interval: int = 100
    ):
        """
        GeminiWrapper uses the GeminiKeyScheduler for managing API keys while supporting the same caching and logging functionality as GPT4oWrapper.
        """
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            num_samples=num_samples,
            text_cache_path=text_cache_path,
            img_cache_path=img_cache_path,
            token_usage_path=token_usage_path,
            max_retries=max_retries,
            sync_interval=sync_interval,
        )
        from utils import GeminiKeyScheduler
        self.scheduler = GeminiKeyScheduler()
        # self.model_name = "gemini-1.5-flash-002"  # Default Gemini model name
        # self.name = "gemini-1.5-flash-002"  # Default Gemini model name
        self.model_name = name
        self.name = name

    def _call_gemini_api(self, prompt, image_path=None):
        """
        Core API call logic with retry and key switching.
        """
        for attempt in range(self.max_retries):
            key = self.scheduler.ask_for_key()
            if not key:
                raise RuntimeError("No valid key available for Gemini inference.")
            try:
                # Set the current key for the Gemini API
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name=self.model_name)
                if image_path:
                    image = Image.open(image_path)
                    response = model.generate_content([prompt, image])
                else:
                    response = model.generate_content([prompt])
                # print(response)
                # if response and hasattr(response, "result"):
                #     result = response.result  # Extract the result part
                #     if hasattr(result, "prompt_feedback"):
                #         prompt_feedback = result.prompt_feedback  # Access prompt_feedback
                #         if hasattr(prompt_feedback, "block_reason"):
                #             block_reason = prompt_feedback.block_reason
                #             print(f"Block reason: {block_reason}")
                #             if block_reason == "PROHIBITED_CONTENT":
                #                 self.scheduler.feedback(success=True)
                #                 return ""
                response_str = str(response)
                if "block_reason: PROHIBITED_CONTENT" in response_str:
                    print("Block reason: PROHIBITED_CONTENT")
                    self.scheduler.feedback(success=True)
                    return ""
                if "RECITATION" in response_str:
                    print("Block reason: RECITATION")
                    self.scheduler.feedback(success=True)
                    return ""

                # print("################")
                if not hasattr(response, "text"):
                    raise RuntimeError("API response missing 'text' attribute.")
                # Provide feedback and return result
                self.scheduler.feedback(success=True)
                return response.text.strip()
            except Exception as e:
                # print(response)
                print(f"[GeminiWrapper] Error during API call with key {key}: {e}")
                # raise e
                time.sleep(1)  # Delay before retry
                self.scheduler.feedback(success=False)

        raise RuntimeError("All retries for Gemini API call failed.")

    def _call_gemini_api_multi(self, user_prompt: str, image_paths: List[str]) -> str:
        """
        核心 API 调用逻辑，具有重试和密钥切换功能，支持多个图片。
        """
        for attempt in range(self.max_retries):
            key = self.scheduler.ask_for_key()
            if not key:
                raise RuntimeError("No valid key available for Gemini inference.")
            try:
                # 设置当前的 Gemini API 密钥
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name=self.model_name)
                if image_paths:
                    # 打开所有图片
                    images = [Image.open(path) for path in image_paths]
                    # 准备输入列表，包含提示和所有图片
                    inputs = [user_prompt] + images
                    # 调用 generate_content 方法
                    response = model.generate_content(inputs)
                else:
                    response = model.generate_content([user_prompt])
                # 将响应转换为字符串以进行检查
                response_str = str(response)
                # 检查是否存在被禁止的内容
                if "block_reason: PROHIBITED_CONTENT" in response_str:
                    print("Block reason: PROHIBITED_CONTENT")
                    self.scheduler.feedback(success=True)
                    return ""
                if "RECITATION" in response_str:
                    print("Block reason: RECITATION")
                    self.scheduler.feedback(success=True)
                    return ""
                # 检查响应是否包含 'text' 属性
                if not hasattr(response, "text"):
                    raise RuntimeError("API response missing 'text' attribute.")
                # 提供反馈并返回结果
                self.scheduler.feedback(success=True)
                return response.text.strip()
            except Exception as e:
                print(f"[GeminiWrapper] Error during API call with key {key}: {e}")
                time.sleep(1)  # 重试前的延迟
                self.scheduler.feedback(success=False)
        raise RuntimeError("All retries for Gemini API call failed.")

    def TextInference(self, queries: List[str]) -> List[str]:
        """
        Handles text-based inference, leveraging caching and the scheduler.
        """
        all_results = []
        for user_prompt in tqdm(queries, desc="Gemini Text Inference"):
            # Cache check and sync
            if self.query_count_text > 0 and self.query_count_text % self.sync_interval == 0:
                self._sync_cache_from_file(self.text_cache_path, self.text_cache)

            self.query_count_text += 1

            # Check cache
            if user_prompt in self.text_cache:
                all_results.append(self.text_cache[user_prompt])
                continue

            # Call Gemini API
            result = self._call_gemini_api(user_prompt)

            # Log result and update cache
            self.text_cache[user_prompt] = result
            self._append_jsonl(self.text_cache_path, user_prompt, result)
            all_results.append(result)

        return all_results

    def VisionInference(self, queries: List[str], image_paths: List[str], image_generator = None) -> List[str]:
        """
        Handles vision-based inference (text + image).
        """
        if len(queries) != len(image_paths):
            raise ValueError("Length of queries and image paths must match for VisionInference.")

        all_results = []
        for user_prompt, img_path in tqdm(zip(queries, image_paths), desc="Gemini Vision Inference"):
            # Cache check and sync
            if self.query_count_img > 0 and self.query_count_img % self.sync_interval == 0:
                self._sync_cache_from_file(self.img_cache_path, self.img_cache)

            self.query_count_img += 1

            # Create cache key
            cache_key = f"{img_path}##{user_prompt}"
            if cache_key in self.img_cache:
                all_results.append(self.img_cache[cache_key])
                continue

            # Call Gemini API
            result = self._call_gemini_api(user_prompt, image_path=img_path)

            # Log result and update cache
            self.img_cache[cache_key] = result
            self._append_jsonl(self.img_cache_path, cache_key, result)
            all_results.append(result)

        return all_results

    def VisionMultiInference(self, queries: List[str], image_paths: List[List[str]], image_generator=None) -> List[str]:
        """
        Handles vision-based inference with multiple images per query.

        Args:
            queries (List[str]): List of user prompts.
            image_paths (List[List[str]]): List of lists of image paths corresponding to each query.
            image_generator: Optional image generator or processor.

        Returns:
            List[str]: List of inference results.
        """
        if len(queries) != len(image_paths):
            raise ValueError("Length of queries and image paths must match for VisionMultiInference.")

        all_results = []
        for user_prompt, img_paths in tqdm(zip(queries, image_paths), desc="Gemini Vision Multi Inference",
                                           total=len(queries)):
            # Cache check and sync
            if self.query_count_img > 0 and self.query_count_img % self.sync_interval == 0:
                self._sync_cache_from_file(self.img_cache_path, self.img_cache)

            self.query_count_img += 1

            # Create a unique cache key using a hash of the sorted image paths and the user prompt
            sorted_img_paths = sorted(img_paths)
            joined_paths = ''.join(sorted_img_paths)
            images_hash = hashlib.md5(joined_paths.encode('utf-8')).hexdigest()
            cache_key = f"{images_hash}##{user_prompt}"

            if cache_key in self.img_cache:
                all_results.append(self.img_cache[cache_key])
                continue

            # Call Gemini API with multiple images
            result = self._call_gemini_api_multi(user_prompt, image_paths=img_paths)

            # Log result and update cache
            self.img_cache[cache_key] = result
            self._append_jsonl(self.img_cache_path, cache_key, result)
            all_results.append(result)

        return all_results
class TestWrapper(ModelWrapperBase):
    name = "test"
    def __init__(self):
        super().__init__()
        self.name = "test"
        self.batch_size = 8
    def VisionInference(self, queries, image_paths, image_generator = None):
        # print("Running vision inference for multiple queries with batch size:", self.batch_size)
        # print("Queries:", queries[:5])
        # print("Image paths:", image_paths[:5])
        return ["A"] * len(queries)
    def TextInference(self, queries):
        # print("Running text inference for multiple queries with batch size:", self.batch_size)
        # print("Queries:", queries[:5])
        return ["""def first_repeated_char(str1): 
    for index,c in enumerate(str1): 
        if str1[:index+1].count(c) > 1: 
            return c
"""] * len(queries)
    def VisionMultiInference(self, queries, image_paths, image_generator = None):
        # print("Running vision inference for multiple queries with batch size:", self.batch_size)
        # print("Queries:", queries[:5])
        # print("Image paths:", image_paths[:5])
        return ["A"] * len(queries)


class MegaBenchModelWrapper(ModelWrapperBase):
    def __init__(self, model_name:str, model_class_name: str, model_hf_name: str, batch_size: int = 128):
        super().__init__()
        self.name = model_name
        self.batch_size = batch_size
        try:
            model_module = importlib.import_module('model_fromMegaBench')
            model_class = getattr(model_module, model_class_name)
        except (ModuleNotFoundError, AttributeError):
            raise ValueError(f"Model wrapper class '{model_class_name}' not found in 'model_fromMegaBench' module.")
        self.model = model_class(model=model_hf_name, max_num_image=10)

    def VisionInference(
            self,
            queries: List[str],
            image_paths: List[str],
            image_generator=None
    ) -> List[str]:
        """
        Perform vision-related inference using the underlying model.

        Args:
            queries (List[str]): List of vision-related queries.
            image_paths (List[str]): List of corresponding image file paths (one per query).
            image_generator (Optional): Custom image generator (not used in this example).

        Returns:
            List[str]: List of model responses corresponding to each query.
        """
        # 构造一个符合 BaseModel 及其子类需要的完整 query_data 结构
        # 包含 global_images, example_info 等字段，哪怕是空的也要写上
        query_data = {
            "global_description": "",
            "global_images": [],  # 空列表
            "example_info": {
                "example_text": "",  # 这里也可以根据需求插入示例文本
                "image_paths": [],  # 这里可以放示例图片路径，如果没有就留空
            },
            "queries": []
        }

        # 将外部传入的 query 与 image_path 组装到 queries 里
        for query, img_path in zip(queries, image_paths):
            query_data["queries"].append({
                "query_text": "<image>" + query,
                "image_paths": [img_path],
                # 如果有正确答案相关信息，需要传 "correct_answer" 字段
                # "correct_answer": ...
            })

        responses = []
        # 按照 batch_size 对 queries 分批推理
        for i in range(0, len(query_data["queries"]), self.batch_size):
            batch_queries = query_data["queries"][i: i + self.batch_size]
            query_data_batch = {
                "global_description": query_data["global_description"],
                "global_images": query_data["global_images"],
                "example_info": query_data["example_info"],
                "queries": batch_queries
            }

            # 调用底层 model 的 query 方法
            batch_responses = self.model.query("VisionInference", query_data_batch)
            for response_item in batch_responses:
                # 从返回结果中提取具体的回答
                responses.append(response_item.get("response", ""))

        return responses

    def VisionMultiInference(
            self,
            queries: List[str],
            image_paths: List[List[str]],
            image_generator=None
    ) -> List[str]:
        """
        使用底层模型执行与视觉相关的推理，每个查询可以包含多个图像。

        Args:
            queries (List[str]): 视觉相关的查询列表。
            image_paths (List[List[str]]): 对应每个查询的图像文件路径列表的列表。
            image_generator (Optional): 自定义图像生成器（在此示例中未使用）。

        Returns:
            List[str]: 与每个查询对应的模型响应列表。
        """
        # 确保 queries 和 image_paths 的长度匹配
        if len(queries) != len(image_paths):
            raise ValueError("在 VisionMultiInference 中，queries 和 image_paths 的长度必须匹配。")

        # 构造一个符合 BaseModel 及其子类需要的完整 query_data 结构
        # 包含 global_images, example_info 等字段，哪怕是空的也要写上
        query_data = {
            "global_description": "",
            "global_images": [],  # 空列表
            "example_info": {
                "example_text": "",  # 这里也可以根据需求插入示例文本
                "image_paths": [],  # 这里可以放示例图片路径，如果没有就留空
            },
            "queries": []
        }

        # 将外部传入的 queries 和 image_paths 组装到 queries 列表中
        for query, imgs_paths in zip(queries, image_paths):
            query_data["queries"].append({
                "query_text": "<image>" * len(imgs_paths) + query,
                "image_paths": imgs_paths,
                # 如果有正确答案相关信息，需要传 "correct_answer" 字段
                # "correct_answer": ...
            })

        responses = []
        # 按照 batch_size 对 queries 分批推理
        for i in range(0, len(query_data["queries"]), self.batch_size):
            batch_queries = query_data["queries"][i: i + self.batch_size]
            query_data_batch = {
                "global_description": query_data["global_description"],
                "global_images": query_data["global_images"],
                "example_info": query_data["example_info"],
                "queries": batch_queries
            }

            # 调用底层模型的 query 方法
            batch_responses = self.model.query("VisionInference", query_data_batch)
            for response_item in batch_responses:
                # 从返回结果中提取具体的回答
                responses.append(response_item.get("response", ""))

        return responses

    def TextInference(self, queries: List[str]) -> List[str]:
        """
        Perform text-only inference using the underlying model.

        Args:
            queries (List[str]): List of text-based queries.

        Returns:
            List[str]: List of model responses.
        """
        # 与 VisionInference 类似，只是这里不需要给出图像路径
        query_data = {
            "global_description": "",
            "global_images": [],
            "example_info": {
                "example_text": "",
                "image_paths": [],
            },
            "queries": []
        }

        # 只在文本查询中，image_paths 为空
        for query in queries:
            query_data["queries"].append({
                "query_text": query,
                "image_paths": [],
                # 如果需要正确答案字段，也可以填充
                # "correct_answer": ...
            })

        responses = []
        # 同样分批推理
        for i in range(0, len(query_data["queries"]), self.batch_size):
            batch_queries = query_data["queries"][i: i + self.batch_size]
            query_data_batch = {
                "global_description": query_data["global_description"],
                "global_images": query_data["global_images"],
                "example_info": query_data["example_info"],
                "queries": batch_queries
            }

            batch_responses = self.model.query("TextInference", query_data_batch)
            for response_item in batch_responses:
                responses.append(response_item.get("response", ""))

        return responses

def load_modelObject(model_name, input_method = "text"): # text or img
    if model_name == "Gemini_Flash":
        model = GeminiWrapper(name="gemini-1.5-flash-002")
        return model
    elif model_name == "Gemini_Thinking":
        model = GeminiWrapper(name="gemini-2.0-flash-thinking-exp-1219")
        return model
    elif model_name == "GPT4o": # name = "gpt-4o"
        model = GPT4oWrapper()
        return model
    elif model_name == "Qwen2_VL_72B":
        # model = MegaBenchModelWrapper(model_name="InternVL2_8B", model_class_name="InternVL",
        #                               model_hf_name="OpenGVLab/InternVL2-Llama3-76B")
        # return model
        model = QWen2VLWrapper(model_name="Qwen2-VL-72B-Instruct", model_path = "Qwen/Qwen2-VL-72B-Instruct",
                               batch_size=1 if input_method == "img" else 8)
        return model
        pass
    elif model_name == "Qwen2_VL_7B":
        model = MegaBenchModelWrapper(model_name="Qwen2-VL-7B-Instruct", model_class_name="Qwen2VL",
                                      model_hf_name="Qwen/Qwen2-VL-7B-Instruct")
        # model = MegaBenchModelWrapper(model_name="InternVL2_8B", model_class_name="InternVL",
        #                               model_hf_name="OpenGVLab/InternVL2-Llama3-76B")
        # model = QWen2VLWrapper(model_name="Qwen2-VL-7B-Instruct", batch_size=1 if input_method == "img" else 8)
        return model
    elif model_name == "Qwen2_VL_2B":
        # model = MegaBenchModelWrapper(model_name="Qwen2-VL-2B-Instruct", model_class_name="Qwen2VL",
        #                               model_hf_name="Qwen/Qwen2-VL-2B-Instruct")
        model = QWen2VLWrapper(model_name="Qwen2-VL-2B-Instruct", model_path = "Qwen/Qwen2-VL-2B-Instruct",
                               batch_size=1 if input_method == "img" else 1)
        return model
    elif model_name == "QVQ_72B_Preview":
        model = QWen2VLWrapper(model_name="QVQ_72B_Preview", model_path = "Qwen/QVQ-72B-Preview",  batch_size=1 if
        input_method == "img" else 1)
        return model
    elif model_name == "InternVL2-Llama3-76B":
        model = MegaBenchModelWrapper(model_name="InternVL2_8B", model_class_name="InternVL",
                                      model_hf_name="OpenGVLab/InternVL2-Llama3-76B")
        return model
    elif model_name == "InternVL2-8B":
        model = MegaBenchModelWrapper(model_name="InternVL2_8B", model_class_name="InternVL",
                                      model_hf_name="OpenGVLab/InternVL2-8B")
        return model
    elif model_name == "Llava_OneVision_72B":
        raise NotImplementedError("Llava_OneVision_72B is not supported in this version.")
    elif model_name == "Llava_OneVision_7B":
        raise NotImplementedError("Llava_OneVision_7B is not supported in this version.")
    elif model_name == "Pixtral-12B":
        model = MegaBenchModelWrapper(model_name="Pixtral-12B", model_class_name="Pixtral",
                                      model_hf_name="mistralai/Pixtral-12B-2409")
        return model
    elif model_name == "Phi_3_5_vision":
        model = MegaBenchModelWrapper(model_name="Phi-3.5-vision", model_class_name="Phi3v",
                                      model_hf_name="microsoft/Phi-3.5-vision-instruct")
        return model
    elif model_name == "Test":
        return TestWrapper()
    else:
        raise ValueError(f"Model {model_name} not found.")


def test_model(model_name, cache_path="test_cache.json"):
    def load_cache(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return {}
    def save_cache(cache_path, cache):
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    cache = load_cache(cache_path)
    if model_name in cache:
        print(f"Results for model {model_name} are already cached.")
        return cache[model_name]

    # Define image paths
    image1 = "/home/zhiheng/WordAsPixel/data/wikiss_subset_old/1002510.jpg"
    image2 = "/home/zhiheng/WordAsPixel/data/wikiss_subset_old/1003950.jpg"

    # Define queries
    queries_single = ["Describe the content of the image."]
    queries_batch = [
        "What is shown in the image?",
        "What does the image depict?",
        "Provide instructions based on the image content.",
        "Identify key elements in the image.",
        "Explain the context of the image.",
        "What could be the purpose of the image?",
        "Summarize the visual content.",
        "Describe the setting in the image."
    ]

    # Vision Inference Tests
    model = load_modelObject(model_name, input_method="img")
    vision_result_single = model.VisionInference(queries_single, [image1])
    vision_result_batch = model.VisionInference(queries_batch, [image1] * len(queries_batch))

    # Vision Multi Inference Tests
    queries_multi = [
        "Compare the two images and describe the differences.",
        "Analyze the common elements between the two images.",
        "Provide a detailed description of both images."
    ]
    image_paths_multi = [[image1, image2]] * len(queries_multi)
    vision_multi_result_single = model.VisionMultiInference(queries_multi, image_paths_multi)

    queries_multi_batch = [
        "Batch comparison and analysis of images.",
        "Evaluate the visual elements in these image sets.",
        "Summarize the relationships across these image pairs."
    ]
    image_paths_multi_batch = [[image1, image2], [image1, image2], [image1, image2]]
    vision_multi_result_batch = model.VisionMultiInference(queries_multi_batch, image_paths_multi_batch)

    del model

    # Text Inference Tests
    model = load_modelObject(model_name, input_method="text")
    text_result_single = model.TextInference(queries_single)
    text_result_batch = model.TextInference(queries_batch)
    del model

    # Save results to cache
    cache[model_name] = {
        "vision_result_single": vision_result_single,
        "vision_result_batch": vision_result_batch,
        "vision_multi_result_single": vision_multi_result_single,
        "vision_multi_result_batch": vision_multi_result_batch,
        "text_result_single": text_result_single,
        "text_result_batch": text_result_batch
    }
    save_cache(cache_path, cache)
    return cache[model_name]

def test_allModel():
    # ["Gemini_Flash", "Gemini_Thinking", "GPT4o",
    # "Qwen2_VL_72B", "Qwen2_VL_7B", "Qwen2_VL_2B", "QVQ_72B_Preview",
    # "InternVL2-Llama3-76B", "InternVL2-8B",
    # "Llava_OneVision_72B", "Llava_OneVision_7B",
    # "Pixtral-12B", "Phi_3_5_vision"]
    model_list = [ "Qwen2_VL_2B",  "GPT4o", "Qwen2_VL_7B", "Phi_3_5_vision", "Gemini_Flash"
        # "Gemini_Flash", "Gemini_Thinking", "GPT4o", "Qwen2_VL_7B", "Qwen2_VL_2B", "InternVL2-8B", "Pixtral-12B", "Phi_3_5_vision"
                   # "InternVL2-Llama3-76B", "Qwen2_VL_72B", "QVQ_72B_Preview",# OOM
                  # "Llava_OneVision_72B", "Llava_OneVision_7B", # Not implemented
                ]
    for model_name in model_list:
        print(f"Testing model: {model_name} ...")
        test_model(model_name)
        print(f"Model {model_name} test passed.\n")

def load_model_input_name(model_name):
    model_input_names = {
        "Gemini_Flash": "gemini-1.5-flash-002",
        "Gemini_Thinking": "gemini-2.0-flash-thinking-exp-1219",
        "GPT4o": "gpt-4o",
        "Qwen2_VL_72B": "Qwen2-VL-72B-Instruct",
        "Qwen2_VL_7B": "Qwen2-VL-7B-Instruct",
        "Qwen2_VL_2B": "Qwen2-VL-2B-Instruct",
        "QVQ_72B_Preview": "QVQ_72B_Preview",
        "InternVL2-Llama3-76B": "InternVL2-Llama3-76B",
        "InternVL2-8B": "InternVL2-8B",
        "Pixtral-12B": "Pixtral-12B",
        "Phi_3_5_vision": "Phi-3.5-vision",
        "Test": "TestWrapper"
    }

    if model_name in model_input_names:
        return model_input_names[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in input mapping.")

if __name__ == "__main__":
    test_allModel()
    # gpt_model = load_modelObject("GPT4o")
    # queries_single = ["What is the answer of 1+1?"]
    # text_result_single = gpt_model.TextInference(queries_single)
    # print(text_result_single)