import os
import sys
import json
import time
import shutil
import textwrap
import evaluate  # Hugging Face evaluate库
import argparse
import importlib
import traceback
import concurrent.futures
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

import numpy as np
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Union
from datasets import load_dataset  # Hugging Face datasets库
from model import ModelWrapperBase
from tqdm import tqdm
from image_generate import ImageGenerator

class DatasetWarperBase(ABC):
    REQUIRED_VARS = ["name", "huggingface_name", "tasks", "image_generator"]

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super().__init__()
        self.cot_flag = cot_flag
        if self.cot_flag:
            self.base_template = (
                "Please first reason step by step, then only provide your final answer after 'Answer:'.\n"
                "Please use the following template:\n"
                "Reasoning:[TODO]\n"
                "Answer:[TODO]"
            )
        else:
            self.base_template = (
                "Please only provide your final answer after 'Answer:'.\n"
                "Please use the following template:\n"
                "Answer:[TODO]"
            )
        self.mode = mode
        self.init_data = init_data
        if from_hf:
            self.PureTextData = self.PureTextData_hf
            self.VisionData = self.VisionDataData_hf
        if init_data:
            if mode == 'text':
                self.data_dict = self.PureTextData()
            elif mode == 'img' or mode == 'semi':
                self.data_dict = self.VisionData()
            else:
                raise NotImplementedError
        else:
            self.data_dict = self.PureTextData()
        self.cache_path = f"./result_cache/"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for var in cls.REQUIRED_VARS:
            if not hasattr(cls, var):
                raise TypeError(f"Class {cls.__name__} is missing required class variable: {var}")

    @abstractmethod
    def PureTextData(self, *args, **kwargs):
        """
        Abstract method for handling pure text data.
        Must be implemented in subclasses.
        """
        pass

    def VisionData(self, vision_type="simple"):
        """
        A function to generate VisionData:
        Format: dict of Dataset {subset_name: [[QueryText, QueryImagePath, Answer, Metric], ...]}
        """
        query_text_template = "Please answer the question in the image."
        DatasetList = self.PureTextData()
        data_dict = self.image_generator.construct_image_dataset(self.name, DatasetList, query_text_template, cot_flag=self.cot_flag)
        return data_dict

    def PureTextData_hf(self):
        """
        Load the TIGER-Lab/PixelWorld dataset and return
        a dict keyed by Subset, with values as list of [Text_Prompt, Answer].
        """
        hf_path = "TIGER-Lab/PixelWorld"
        dataset = load_dataset(hf_path, split="train", trust_remote_code=True)
        dataset = dataset.filter(lambda x: x["Dataset"] == self.name + "Dataset")
        data_dict = {}
        total_count = 0

        for sample in tqdm(dataset):
            subset = sample["Subset"]
            text_prompt = sample["Text_Prompt"]
            answer = sample["Answer"]

            if subset not in data_dict:
                data_dict[subset] = []

            data_dict[subset].append([text_prompt, answer])
            total_count += 1

        print(f"Total examples: {total_count}")
        return data_dict

    def VisionDataData_hf(self):
        """
        Load the TIGER-Lab/PixelWorld dataset and return
        a dict keyed by Subset, with values as a list of [Vision Prompt, local_path(s), Answer].

        Special rule:
        - If self.name is NOT in ["WikiSS_QA", "SlidesVQA"], only use the FIRST image path.
        - Otherwise, use all image paths.

        Also ensure the image(s) is saved locally under 'data/' + img_path if not already.
        """
        hf_path = "TIGER-Lab/PixelWorld"
        dataset = load_dataset(hf_path, split="train", trust_remote_code=True)
        # 根据 Dataset 名进行过滤
        dataset = dataset.filter(lambda x: x["Dataset"] == self.name + "Dataset")

        data_dict = {}
        total_count = 0

        for sample in tqdm(dataset):
            subset = sample["Subset"]
            img_prompt = sample["Img_Prompt"]
            answer = sample["Answer"]

            # 如果这个 subset 还没出现过，就初始化
            if subset not in data_dict:
                data_dict[subset] = []

            # 如果不是 WikiSS_QA 或 SlidesVQA，只用第一个图像
            if self.name not in ["WikiSS_QA", "SlidesVQA"]:
                # 只取第一个路径 + 图像对象
                first_path = sample["Image_Pathes"][0]
                first_img_obj = sample["Images"][0]

                local_path = os.path.join("data", first_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not os.path.exists(local_path):
                    first_img_obj.save(local_path)

                # 这里存的仅是一个字符串
                data_dict[subset].append([img_prompt, local_path, answer])

            else:
                # 使用所有图像路径
                local_paths = []
                for path, img_obj in zip(sample["Image_Pathes"], sample["Images"]):
                    local_path = os.path.join("data", path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    if not os.path.exists(local_path):
                        img_obj.save(local_path)

                    local_paths.append(local_path)

                # 这里存的是一个由多个路径组成的列表
                data_dict[subset].append([img_prompt, local_paths, answer])

            total_count += 1

        print(f"Total examples: {total_count}")
        return data_dict

    @abstractmethod
    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time=True):
        pass

    def InferenceTask(self, model, data_list, task_cache_path, batch_size=8192, tqdm_obj=None, eval_time=False):
        """
        Perform inference with strict sequential processing and batching.

        Args:
            model: The inference model with TextInference and VisionInference methods.
            data_list: List of data samples for inference.
            task_cache_path: Path to the cache file for saving/loading predictions.
            batch_size: Number of samples to process in each batch.
            tqdm_obj: External tqdm object for progress tracking (optional).

        Returns:
            predictions_text: List of predictions.
            references: List of references.
        """
        prompts = [x[0] for x in data_list]
        image_paths = [x[1] for x in data_list] if self.mode == 'img' or self.mode == 'semi' else None
        predictions_text = []
        if os.path.exists(task_cache_path):
            print(f"Loading cached predictions from: {task_cache_path}")
            with open(task_cache_path, 'r', encoding='utf-8') as f:
                predictions_text = json.load(f)
        else:
            if eval_time:
                raise FileNotFoundError(f"No cache found for evaluation: {task_cache_path}")
            print(f"No cache found. Starting inference from the beginning.")
        start_idx = len(predictions_text)
        if tqdm_obj is None:
            tqdm_obj = tqdm(total=len(prompts), initial=start_idx, desc="Inference Progress")

        for batch_start in range(start_idx, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            if self.mode == 'text':
                batch_predictions = model.TextInference(batch_prompts)
            elif (self.mode == 'img' or self.mode == 'semi') and self.name != "WikiSS_QA" and self.name != "SlidesVQA":
                batch_image_paths = image_paths[batch_start:batch_end]
                batch_predictions = model.VisionInference(batch_prompts, batch_image_paths, self.image_generator)
            elif self.mode == 'img' and (self.name == "WikiSS_QA" or self.name == "SlidesVQA"):
                # These two dataset has multiple imgs
                # TODO: check if it has bugs.
                batch_image_paths = image_paths[batch_start:batch_end]
                batch_predictions = model.VisionMultiInference(batch_prompts, batch_image_paths, self.image_generator)
            else:
                print(f"Mode: {self.mode}; Name: {self.name}")
                raise NotImplementedError("Mode not supported")

            # Update predictions and write to cache
            predictions_text.extend(batch_predictions)
            with open(task_cache_path, 'w', encoding='utf-8') as f:
                json.dump(predictions_text, f, ensure_ascii=False, indent=2)

            # Update progress
            tqdm_obj.update(len(batch_prompts))

        tqdm_obj.close()
        return predictions_text

    def FinalScore(self, task_results):
        final_score = np.mean(list(task_results.values()))
        print(f"\n--- {self.name} Evaluation Results ---")
        for task, score in task_results.items():
            print(f"{task}: {score:.4f}")
        print(f"\nFinal {self.name} Score: {final_score:.4f}")
        return final_score

    def InferenceDataset(self, model):
        """
        Abstract method for inference and evaluation.
        Must be implemented in subclasses.
        :param model: The model to be used for inference.
        """
        prompt_name = "CoT" if self.cot_flag else "Base"
        cache_path = f"{self.cache_path}/{model.name}/{self.name}/"
        os.makedirs(cache_path, exist_ok=True)  # 递归创建目录
        for task in self.tasks:
            task_cache_path = os.path.join(cache_path, f"{task}_{self.mode}_{prompt_name}.json")
            self.InferenceTask(model, self.data_dict[task], task_cache_path)

    # @abstractmethod
    def Eval(self, model_name):
        """
        Abstract method for evaluation only.
        Must be implemented in subclasses.
        :param model: The model to be used for inference.
        """
        prompt_name = "CoT" if self.cot_flag else "Base"
        cache_path = f"{self.cache_path}/{model_name}/{self.name}/"
        task_results = {}
        for task in self.tasks:
            task_cache_path = os.path.join(cache_path, f"{task}_{self.mode}_{prompt_name}.json")
            if self.name in ["SuperGLUE"]:
                dataset = load_dataset(self.huggingface_name, task, split="validation", trust_remote_code=True)
            elif self.name in []:
                dataset = load_dataset(self.huggingface_name, task, split="test", trust_remote_code=True)
            elif self.name in ["MMLU", "MMLU-Pro", "GLUE", "ARC", "MBPP", "HumanEval", "MATH", "GSM8K", "GPQA",
                               "TableBench", "WikiSS_QA", "SlidesVQA", "Mathverse"]:
                dataset = None
            else:
                raise NotImplementedError
            predictions_text = self.InferenceTask(None, self.data_dict[task], task_cache_path, eval_time=True)
            if not self.init_data:
                references_text = [x[1] for x in self.data_dict[task]]
            else:
                references_text = [x[1] for x in self.data_dict[task]] if self.mode == 'text' else [x[2] for x in
                                                                                            self.data_dict[task]]
            predictions, references = self.FormatPredNRef(dataset, task, predictions_text, references_text)
            if self.name in ["TableBench"]: # Add Tables into references
                tables = self.GetTables(task)
                results = self.GetResult(task, predictions, references, tables)
            else:
                results = self.GetResult(task, predictions, references)
            if isinstance(results, dict):
                task_score = np.mean(list(results.values()))
            else:
                task_score = results
            task_results[task] = task_score
            print(f"Results for {task}: {results}")
        final_score = self.FinalScore(task_results)
        return task_results, final_score

    @abstractmethod
    def GetResult(self, task, prediction, reference):
        pass

# SuperGLUE: https://huggingface.co/datasets/hgissbkh/super_glue
class SuperGLUEDataset(DatasetWarperBase):
    name = "SuperGLUE"
    huggingface_name = "super_glue"
    tasks = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(SuperGLUEDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Modified prompt templates with:
          1) Explicit task descriptions for each benchmark.
          2) Clear instructions to reason step-by-step (if cot_flag=True)
             and then provide the answer.
          3) Explicitly indicating what the model should output after 'Answer:'.
        """
        # 针对不同任务，指定“答案”的要求（answer_instructions）
        if task == "boolq":
            description = (
                "The BoolQ task requires the model to answer a yes/no question based on a given passage."
            )
            # 明确告诉模型只能输出 True 或 False
            answer_instructions = "Your final answer must be exactly 'True' or 'False'."
            return (
                f"Task: BoolQ\n"
                f"{description}\n"
                f"Question: {sample['question']}\n"
                f"Passage: {sample['passage']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "cb":
            description = (
                "The CommitmentBank (CB) task asks the model to determine whether a given hypothesis is "
                "'Entailment', 'Contradiction', or 'Neutral' based on a provided premise."
            )
            # CB 的答案格式：Entailment / Contradiction / Neutral
            answer_instructions = (
                "Your final answer must be one of: 'Entailment', 'Contradiction', or 'Neutral'."
            )
            return (
                f"Task: CommitmentBank (CB)\n"
                f"{description}\n"
                f"Text: {sample['premise']}\n"
                f"Hypothesis: {sample['hypothesis']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "copa":
            description = (
                "The Choice of Plausible Alternatives (COPA) task asks the model to choose between two "
                "options (1 or 2) to identify the most plausible cause or effect of a given premise."
            )
            question = "What is the cause?" if sample["question"] == "cause" else "What is the effect?"
            # COPA 的答案格式：1 或 2
            answer_instructions = "Your final answer must be exactly '1' or '2'."
            return (
                f"Task: Choice of Plausible Alternatives (COPA)\n"
                f"{description}\n"
                f"Premise: {sample['premise']}\n"
                f"{question}\n"
                f"Option 1: {sample['choice1']}\n"
                f"Option 2: {sample['choice2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "multirc":
            description = (
                "The Multi-Sentence Reading Comprehension (MultiRC) task requires the model to evaluate "
                "whether a candidate answer to a question is correct based on a provided paragraph."
            )
            # MultiRC 的答案格式：True 或 False
            answer_instructions = (
                "Your final answer must be exactly 'True' or 'False'."
            )
            return (
                f"Task: Multi-Sentence Reading Comprehension (MultiRC)\n"
                f"{description}\n"
                f"Paragraph: {sample['paragraph']}\n"
                f"Question: {sample['question']}\n"
                f"Candidate Answer: {sample['answer']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "record":
            description = (
                "The Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) task requires the "
                "model to predict an open-ended answer to a query based on a provided passage."
            )
            # ReCoRD 的答案格式：开放式文本
            answer_instructions = (
                "Your final answer can be an open-ended string (no length limit)."
            )
            return (
                f"Task: ReCoRD\n"
                f"{description}\n"
                f"Passage: {sample['passage']}\n"
                f"Query: {sample['query']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "rte":
            description = (
                "The Recognizing Textual Entailment (RTE) task asks the model to determine whether a "
                "hypothesis is 'Entailment' or 'Not Entailment' based on a provided premise."
            )
            # RTE 的答案格式：Entailment / Not Entailment
            answer_instructions = (
                "Your final answer must be exactly 'Entailment' or 'Not Entailment'."
            )
            return (
                f"Task: Recognizing Textual Entailment (RTE)\n"
                f"{description}\n"
                f"Premise: {sample['premise']}\n"
                f"Hypothesis: {sample['hypothesis']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "wic":
            description = (
                "The Words in Context (WiC) task requires the model to determine if a given word is used "
                "with the same meaning in two different sentences."
            )
            # WiC 的答案格式：Yes / No
            answer_instructions = (
                "Your final answer must be exactly 'Yes' or 'No'."
            )
            return (
                f"Task: Words in Context (WiC)\n"
                f"{description}\n"
                f"Sentence 1: {sample['sentence1']}\n"
                f"Sentence 2: {sample['sentence2']}\n"
                f"Word: {sample['word']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "wsc":
            description = (
                "The Winograd Schema Challenge (WSC) requires the model to determine whether a given "
                "pronoun refers to a specific noun in the context of a sentence."
            )
            # WSC 的答案格式：Yes / No
            answer_instructions = (
                "Your final answer must be exactly 'Yes' or 'No'."
            )
            return (
                f"Task: Winograd Schema Challenge (WSC)\n"
                f"{description}\n"
                f"Text: {sample['text']}\n"
                f"Does the pronoun '{sample['span2_text']}' refer to '{sample['span1_text']}'?\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

    def parse_output(self, task, output):
        """
        Parses the model output. Extracts and retains the content after 'Answer:' before processing.
        Returns an integer (or string for open-ended tasks like ReCoRD).
        -1 if unable to parse or conflicting answers.
        """

        # Extract the content after 'Answer:'
        if "answer:" in output.lower():
            output = output.split("Answer:")[-1].strip()

        # 下面的逻辑可根据需要扩展或修改
        if task == "boolq":
            # True or False
            if "true" in output.lower() and "false" in output.lower():
                return -1
            return 1 if "true" in output.lower() else (0 if "false" in output.lower() else -1)

        elif task == "cb":
            # Entailment -> 0, Contradiction -> 1, Neutral -> 2
            low = output.lower()
            has_entail = "entailment" in low
            has_contra = "contradiction" in low
            has_neutral = "neutral" in low
            # 如果有多个不同答案，返回 -1
            if sum([has_entail, has_contra, has_neutral]) != 1:
                return -1
            return 0 if has_entail else (1 if has_contra else 2)

        elif task == "copa":
            # 1 or 2
            if "1" in output and "2" in output:
                return -1
            return 0 if "1" in output else (1 if "2" in output else -1)

        elif task == "multirc":
            # True or False
            if "true" in output.lower() and "false" in output.lower():
                return -1
            return 1 if "true" in output.lower() else (0 if "false" in output.lower() else -1)

        elif task == "record":
            # Open-ended answer string
            return output

        elif task == "rte":
            # Entailment -> 1, Not Entailment -> 0
            low = output.lower()
            # if "entailment" in low and "not entailment" in low:
            #     return -1
            return 1 if "not entailment" in low else (0 if "entailment" in low else -1)

        elif task in ["wic", "wsc"]:
            # Yes -> 1, No -> 0
            low = output.lower()
            if "yes" in low and "no" in low:
                return -1
            return 1 if "yes" in low else (0 if "no" in low else -1)

        return 0

    def PureTextData(self) -> Dict[str, List[List[Union[str, int, list]]]]:
        """
        A function to generate PureTextData.

        Output Format:
        dict of Dataset {subset_name:[ [QueryText, Answer], ... ]}

        subset_name 对应每个任务名称，如"boolq"
        QueryText 为prompt
        Answer 为真实标签
        Metric 为任务相应的评估指标(字符串或字符串列表)
        """
        data_dict = {}
        for task in self.tasks:
            dataset = load_dataset("super_glue", task, split="validation", trust_remote_code=True)
            # print(task, dataset)
            data_list = []
            for sample in dataset:
                # print(sample)
                prompt = self.format_prompt(task, sample)
                answer = sample["label"] if task != "record" else sample["answers"]
                data_list.append([prompt, answer])
            data_dict[task] = data_list
        # exit(0)
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            if task == "multirc":
                idx_info = dataset[idx]["idx"]
                predictions.append({"idx": idx_info, "prediction": pred})
            elif task == "record":
                idx_info = dataset[idx]["idx"]
                predictions.append({"idx": idx_info, "prediction_text": pred})
            else:
                predictions.append(pred)
        if task == "record":
            for idx, ref_label in enumerate(references_text):  # references_raw 是原始参考答案
                idx_info = dataset[idx]["idx"]  # 从 dataset 提取 idx 信息
                references.append({"idx": idx_info, "answers": ref_label})
        else:
            references = references_text
        return predictions, references

    def GetResult(self, task, prediction, reference):
        # TODO, need check
        metric = evaluate.load(self.huggingface_name, task)
        import random
        if task == "multirc":
            for i in prediction:
                i['prediction'] = i['prediction'] if i['prediction']!= -1 else random.choice([0, 1])
        results = metric.compute(predictions=prediction, references=reference)
        return results

# GLUE: https://huggingface.co/datasets/nyu-mll/glue
class GLUEDataset(DatasetWarperBase):
    """
    A dataset wrapper class for the GLUE benchmark, following the same structure
    and style as the SuperGLUE dataset wrapper provided.
    """
    name = "GLUE"
    huggingface_name = "nyu-mll/glue"
    # GLUE tasks. The standard set includes:
    # ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli', 'ax'].
    # Here, we'll use the main tasks. 'ax' is often for diagnostic test only.
    tasks = [
        "cola",  # 0: unacceptable, 1: acceptable
        "sst2",  # 0: negative, 1: positive
        "mrpc",  # 0: not paraphrase, 1: paraphrase
        "qqp",  # 0: not duplicate, 1: duplicate
        "stsb",  # regression [0.0, 5.0]
        "mnli",  # labels: 'entailment', 'neutral', 'contradiction'
        "qnli",  # labels: 'entailment', 'not_entailment'
        "rte",  # labels: 'entailment', 'not_entailment'
        "wnli"  # labels: 'entailment', 'not_entailment'
    ]
    image_generator = ImageGenerator()  # kept for consistency with the original structure

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        """
        :param cot_flag: Whether to include chain-of-thought reasoning in the prompt
        :param mode: 'text' or other mode of data representation
        """
        super(GLUEDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Creates prompt strings for each GLUE task.

        1) Includes a brief task-specific description.
        2) Provides clear instructions regarding the output format.
        3) If cot_flag=True, the base_template may invite step-by-step reasoning.
        """
        # print(task, sample)
        if task == "cola":
            description = (
                "The CoLA (Corpus of Linguistic Acceptability) task requires the model to decide "
                "whether a given sentence is grammatically acceptable or not."
            )
            # Label: 1 = acceptable, 0 = unacceptable
            answer_instructions = (
                "Your final answer must be exactly 'Acceptable' or 'Unacceptable'."
            )
            return (
                f"Task: CoLA\n"
                f"{description}\n"
                f"Sentence: {sample['sentence']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "sst2":
            description = (
                "The SST-2 (Stanford Sentiment Treebank) task requires the model to classify a sentence "
                "as having a positive or negative sentiment."
            )
            # Label: 1 = positive, 0 = negative
            answer_instructions = (
                "Your final answer must be exactly 'Positive' or 'Negative'."
            )
            return (
                f"Task: SST-2\n"
                f"{description}\n"
                f"Sentence: {sample['sentence']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "mrpc":
            description = (
                "The MRPC (Microsoft Research Paraphrase Corpus) task requires the model to determine "
                "whether two sentences are paraphrases of each other."
            )
            # Label: 1 = paraphrase, 0 = not paraphrase
            answer_instructions = (
                "Your final answer must be exactly 'Paraphrase' or 'Not Paraphrase'."
            )
            return (
                f"Task: MRPC\n"
                f"{description}\n"
                f"Sentence 1: {sample['sentence1']}\n"
                f"Sentence 2: {sample['sentence2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "qqp":
            description = (
                "The QQP (Quora Question Pairs) task requires the model to determine whether two "
                "questions are duplicates of each other."
            )
            # Label: 1 = duplicate, 0 = not duplicate
            answer_instructions = (
                "Your final answer must be exactly 'Duplicate' or 'Not Duplicate'."
            )
            return (
                f"Task: QQP\n"
                f"{description}\n"
                f"Question 1: {sample['question1']}\n"
                f"Question 2: {sample['question2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "stsb":
            description = (
                "The STS-B (Semantic Textual Similarity Benchmark) task requires the model to assign "
                "a similarity score (floating point) between 0 and 5 for two given sentences."
            )
            # Label: floating point [0, 5]
            answer_instructions = (
                "Your final answer must be a floating point number between 0.0 and 5.0."
            )
            return (
                f"Task: STS-B\n"
                f"{description}\n"
                f"Sentence 1: {sample['sentence1']}\n"
                f"Sentence 2: {sample['sentence2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "mnli":
            description = (
                "The MNLI (Multi-Genre Natural Language Inference) task requires the model to predict "
                "whether a premise entails, contradicts, or is neutral toward a given hypothesis."
            )
            # Label: 'entailment', 'neutral', 'contradiction'
            answer_instructions = (
                "Your final answer must be one of: 'Entailment', 'Contradiction', or 'Neutral'."
            )
            return (
                f"Task: MNLI\n"
                f"{description}\n"
                f"Premise: {sample['premise']}\n"
                f"Hypothesis: {sample['hypothesis']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "qnli":
            description = (
                "The QNLI (Question Natural Language Inference) task requires the model to determine "
                "whether the answer to a question is entailed by a given passage."
            )
            # Label: 'entailment', 'not_entailment'
            answer_instructions = (
                "Your final answer must be exactly 'Entailment' or 'Not Entailment'."
            )
            return (
                f"Task: QNLI\n"
                f"{description}\n"
                f"Question: {sample['question']}\n"
                f"Passage: {sample['sentence']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "rte":
            description = (
                "The RTE (Recognizing Textual Entailment) task requires the model to determine whether "
                "a hypothesis is entailed by a given premise."
            )
            # Label: 'entailment', 'not_entailment'
            answer_instructions = (
                "Your final answer must be exactly 'Entailment' or 'Not Entailment'."
            )
            return (
                f"Task: RTE\n"
                f"{description}\n"
                f"Premise: {sample['sentence1']}\n"
                f"Hypothesis: {sample['sentence2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )

        elif task == "wnli":
            description = (
                "The WNLI (Winograd NLI) task requires the model to determine "
                "whether a given pronoun resolution is entailed by the sentence."
            )
            # Label: 'entailment', 'not_entailment'
            answer_instructions = (
                "Your final answer must be exactly 'Entailment' or 'Not Entailment'."
            )
            return (
                f"Task: WNLI\n"
                f"{description}\n"
                f"Premise: {sample['sentence1']}\n"
                f"Hypothesis: {sample['sentence2']}\n"
                f"{answer_instructions}\n"
                f"{self.base_template}"
            )
        else:
            raise NotImplementedError

    def parse_output(self, task, output):
        # print()
        """
        Parses the model output. Extracts and retains the content after 'Answer:' before processing.
        Returns an integer, float, or string based on the task:
          -1 if unable to parse or conflicting answers.
        """
        # 1) 提取 "Answer:" 后的内容
        if "answer:" in output.lower():
            output = output.split("Answer:")[-1].strip()

        low = output.lower()

        if task == "cola":
            # Acceptable -> 1, Unacceptable -> 0
            if "unacceptable" in low:
                return 0
            elif "acceptable" in low:
                return 1
            else:
                return -1

        elif task == "sst2":
            # Positive -> 1, Negative -> 0
            if "negative" in low:
                return 0
            elif "positive" in low:
                return 1
            else:
                return -1

        elif task == "mrpc":
            # Paraphrase -> 1, Not Paraphrase -> 0
            if "not paraphrase" in low:
                return 0
            elif "paraphrase" in low:
                return 1
            else:
                return -1

        elif task == "qqp":
            # Duplicate -> 1, Not Duplicate -> 0
            if "not duplicate" in low:
                return 0
            elif "duplicate" in low:
                return 1
            else:
                return -1

        elif task == "stsb":
            # Expect a floating point number between 0 and 5
            import re
            match = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if len(match) == 1:
                val = float(match[0])
                if 0.0 <= val <= 5.0:
                    return val
            return -1

        elif task == "mnli":
            # 'entailment' -> 0, 'contradiction' -> 1, 'neutral' -> 2
            # 注意没有 "not entailment" 这种冲突，所以原逻辑可以
            has_entail = "entailment" in low
            has_contra = "contradiction" in low
            has_neutral = "neutral" in low
            if sum([has_entail, has_contra, has_neutral]) != 1:
                return -1
            return 0 if has_entail else (2 if has_contra else 1)

        elif task == "wnli":
            # 'entailment' -> 1, 'not entailment' -> 0
            # 如果字符串中包含 "not entailment" 也会包含 "entailment"
            if "not entailment" in low:
                return 0
            elif "entailment" in low:
                return 1
            else:
                return -1

        elif task == "qnli" or task == "rte":
            # 'not entailment' -> 1, 'not entailment' -> 0
            # 如果字符串中包含 "not entailment" 也会包含 "entailment"
            if "not entailment" in low:
                return 1
            elif "entailment" in low:
                return 0
            else:
                return -1
        return -1

    def PureTextData(self) -> Dict[str, List[List[Union[str, int, list]]]]:
        """
        A function to generate PureTextData for each GLUE task.

        Output Format:
        dict of Dataset {subset_name:[ [QueryText, Answer], ... ]}

        subset_name corresponds to each GLUE task name, e.g. "cola".
        QueryText is the prompt.
        Answer is the ground-truth label for classification or regression.
        """
        data_dict = {}
        for task in self.tasks:
            # For MNLI, we have matched and mismatched splits. Typically we do "validation_matched".
            # For simplicity, we just pick "validation" or the commonly used split for the task.
            # Some tasks have a single "validation" set, others might have special splits.
            # We'll do a small conditional check:
            split_name = "validation"
            if task == "mnli":
                split_name = "validation_matched"  # using matched as a default
            dataset = load_dataset(self.huggingface_name, task, split=split_name, trust_remote_code=True)

            data_list = []
            for sample in dataset:
                prompt = self.format_prompt(task, sample)
                # For STS-B, label is a float. For classification tasks, label is int.
                # We'll directly store the label as is.
                # The label key often is 'label', except certain tasks have different field names.
                # But HF's GLUE usually standardizes it to 'label'.
                answer = sample["label"]
                data_list.append([prompt, answer])
            data_dict[task] = data_list
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time=True):
        """
        Convert raw text outputs into the format needed for the metric computation.
        Return predictions, references in the correct structure.
        """
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            parsed_pred = self.parse_output(task, p_out)
            predictions.append(parsed_pred)
        references = references_text
        import random
        predictions = [random.choice(list(set(references))) if p == -1 else p for p in predictions]
        return predictions, references

    def GetResult(self, task, predictions, references):
        """
        Compute the evaluation metrics for the given GLUE-like task using Hugging Face evaluate.
        Return a dict, e.g., {"accuracy": 0.900, "f1": 0.915}, {"spearmanr": 0.89, "pearsonr": 0.91}, etc.

        Parameters
        ----------
        task : str
            任务名称，比如 "cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"。
        predictions : List
            模型的预测结果。
        references : List
            真实标签。

        Returns
        -------
        results : dict
            不同任务返回的指标项略有差异。例如:
            - "cola" -> {"matthews_correlation": 0.52}
            - "sst2" -> {"accuracy": 0.91}
            - "mrpc"/"qqp" -> {"accuracy": 0.85, "f1": 0.88}
            - "stsb" -> {"pearsonr": 0.90, "spearmanr": 0.88}
            - 其他分类任务 -> {"accuracy": x.xx}
        """
        # 1. CoLA: 语言学可接受性, 使用 matthews_correlation
        if task == "cola":
            metric = evaluate.load("matthews_correlation")
            result = metric.compute(predictions=predictions, references=references)
            # 返回值形如 {"matthews_correlation": 0.xx}
            return result
        # 2. MRPC/QQP: 通常希望同时返回 accuracy 和 f1
        elif task in ["mrpc", "qqp"]:
            # 分别计算 accuracy 和 f1，然后合并到一个 dict
            acc_metric = evaluate.load("accuracy")
            # f1_metric = evaluate.load("f1")
            print(predictions[:10], references[:10])
            acc_res = acc_metric.compute(predictions=predictions, references=references)
            # f1_res = f1_metric.compute(predictions=predictions, references=references)
            from sklearn.metrics import f1_score
            f1_value = f1_score(references, predictions, average='macro')
            # 合并两个字典
            # print(acc_res, f1_value)
            # exit(0)
            merged_res = {"accuracy": acc_res['accuracy'], "f1": f1_value}
            # 例如 {"accuracy": 0.85, "f1": 0.88}
            return merged_res
        # 3. STS-B: 回归相似度, 通常用 pearsonr + spearmanr
        elif task == "stsb":
            pearson_metric = evaluate.load("pearsonr")
            spearman_metric = evaluate.load("spearmanr")
            pearson_res = pearson_metric.compute(predictions=predictions, references=references)
            spearman_res = spearman_metric.compute(predictions=predictions, references=references)
            # 返回 {"pearsonr": ..., "spearmanr": ...}
            merged_res = {**pearson_res, **spearman_res}
            return merged_res
        # 4. 其他主要是分类任务, 使用 accuracy 即可 (SST-2, MNLI, QNLI, RTE, WNLI ...)
        #    如果你的 MNLI 拆成 matched/mismatched，请自行处理
        elif task in ["sst2", "mnli", "qnli", "rte", "wnli"]:
            metric = evaluate.load("accuracy")
            result = metric.compute(predictions=predictions, references=references)
            # 返回形如 {"accuracy": x.xx}
            return result
        else:
            # 如果是未知任务，或者你有自定义任务，可以在这里做扩展
            raise ValueError(f"Unknown task: {task}")

# MMLU: https://huggingface.co/datasets/cais/mmlu
class MMLUDataset(DatasetWarperBase):
    name = "MMLU"
    huggingface_name = "cais/mmlu"
    categories = {
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other (business, health, misc.)": ["other", "business", "health"],
    }
    subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }
    tasks = list(subcategories.keys())
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(MMLUDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Formats the prompt for MMLU tasks with optional Chain-of-Thought.

        Args:
            task (str): The specific MMLU task name.
            sample (dict): A dictionary containing:
                - 'question' (str): The question to be answered.
                - 'choices' (List[str]): A list of possible answer choices.

        Returns:
            str: The formatted prompt string.
        """
        # 简要说明
        description = (
            f"The {task} task requires the model to select the most appropriate choice "
            f"(A, B, C, or D) for a given question."
        )
        instructions = "Your final answer must be exactly one of: 'A', 'B', 'C', or 'D'."

        # 格式化选项
        choices_formatted = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample['choices'])
        ])

        return (
            f"Task: {task}\n"
            f"{description}\n"
            f"Question: {sample['question']}\n"
            f"Choices:\n{choices_formatted}\n"
            f"{instructions}\n"
            f"{self.base_template}"
        )

    def parse_output(self, task, output):
        """
        Parses the model output for MMLU tasks.
        1) 提取并仅保留 'Answer:' 后面的内容。
        2) 将结果转成 a, b, c, d 等小写后映射为索引。

        Args:
            task (str): The specific MMLU task name.
            output (str): The raw output string from the model.

        Returns:
            int: The index of the chosen option (0 for A, 1 for B, etc.), or -1 if invalid.
        """
        lower_output = output.lower()
        if "answer:" in lower_output:
            lower_output = lower_output.split("answer:")[-1].strip()
        lower_output = lower_output.strip()
        if lower_output in ["a", "b", "c", "d"]:
            return ord(lower_output) - ord("a")
        return -1

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        for task in tqdm(self.tasks):
            dataset = load_dataset(self.huggingface_name, task, split="test", trust_remote_code=True)
            data_list = []
            for sample in dataset:
                # print(sample)
                prompt = self.format_prompt(task, sample)
                answer = sample["answer"]
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
        references = references_text
        return predictions, references

    def GetResult(self, task, prediction, reference):
        metric = evaluate.load("accuracy")
        results = metric.compute(predictions=prediction, references=reference)
        return results

# MMLU Pro: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
class MMLUProDataset(DatasetWarperBase):
    name = "MMLU-Pro"
    huggingface_name = "TIGER-Lab/MMLU-Pro"
    tasks = ['health', 'history', 'chemistry', 'computer science', 'biology', 'engineering', 'other', 'economics', 'math', 'psychology', 'philosophy', 'business', 'law', 'physics']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(MMLUProDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Formats the prompt for MMLU tasks with optional Chain-of-Thought.

        Args:
            task (str): The specific MMLU task name.
            sample (dict): A dictionary containing:
                - 'question' (str): The question to be answered.
                - 'choices' (List[str]): A list of possible answer choices.

        Returns:
            str: The formatted prompt string.
        """
        # 简要说明
        description = (
            f"The {task} task requires the model to select the most appropriate choice "
            f"(A, B, C, D, E, F, G, H, I or J) for a given question."
        )
        instructions = "Your final answer must be exactly one of: 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', or 'J'."

        # 格式化选项
        choices_formatted = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample['options'])
        ])

        return (
            f"Task: {task}\n"
            f"{description}\n"
            f"Question: {sample['question']}\n"
            f"Choices:\n{choices_formatted}\n"
            f"{instructions}\n"
            f"{self.base_template}"
        )

    def parse_output(self, task, output):
        """
        Parses the model output for MMLU tasks.
        1) 提取并仅保留 'Answer:' 后面的内容。
        2) 将结果转成 a, b, c, d 等小写后映射为索引。

        Args:
            task (str): The specific MMLU task name.
            output (str): The raw output string from the model.

        Returns:
            int: The index of the chosen option (0 for A, 1 for B, etc.), or -1 if invalid.
        """
        lower_output = output.lower()
        if "answer:" in lower_output:
            lower_output = lower_output.split("answer:")[-1].strip()
        lower_output = lower_output.strip()
        # 10 opinions: a, b, c, d, e, f, g, h, i, j
        if lower_output in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]:
            return ord(lower_output) - ord("a")
        return -1

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        # Load the test split of the dataset
        dataset = load_dataset(self.huggingface_name, split="test", trust_remote_code=True)
        print(dataset)
        print(set(dataset["category"]))
        for task in tqdm(self.tasks):
            # dataset = load_dataset(self.huggingface_name, split="test", trust_remote_code=True)
            # Select the category = task
            dataset_task = dataset.filter(lambda x: x["category"] == task)
            data_list = []
            for sample in dataset_task:
                # print(sample)
                prompt = self.format_prompt(task, sample)
                answer = sample["answer"]
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
        # Reference = ord(Reference) - "A"
        references = [ord(ref) - ord("A") for ref in references_text]
        return predictions, references

    def GetResult(self, task, prediction, reference):
        metric = evaluate.load("accuracy")
        results = metric.compute(predictions=prediction, references=reference)
        return results

# GSM8K: https://huggingface.co/datasets/openai/gsm8k
class GSM8KDataset(DatasetWarperBase):
    name = "GSM8K"
    huggingface_name = "openai/gsm8k"
    tasks = ["main"]
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(GSM8KDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Formats the prompt for GSM8K tasks, ensuring the answer is a valid integer or float.

        Args:
            task (str): The specific GSM8K task name.
            sample (dict): A dictionary containing:
                - 'question' (str): The question to be answered.

        Returns:
            str: The formatted prompt string.
        """
        # Brief description
        description = (
            f"The {task} task requires the model to solve the given math problem and provide an answer "
            f"that is either an integer or a float."
        )
        instructions = (
            "Your final answer must be a numerical value, either an integer or a float, "
            "and it should not include any units or additional text. Examples of valid answers: 13, 7.5."
        )

        return (
            f"Task: {task}\n"
            f"{description}\n"
            f"Question: {sample['question']}\n"
            f"{instructions}\n"
            f"{self.base_template}"
        )

    def parse_output(self, task, output):
        """
        Parses the model output for MMLU tasks.
        1) 提取并仅保留 'Answer:' 后面的内容。
        2) 将结果转成 a, b, c, d 等小写后映射为索引。

        Args:
            task (str): The specific MMLU task name.
            output (str): The raw output string from the model.

        Returns:
            int: The index of the chosen option (0 for A, 1 for B, etc.), or -1 if invalid.
        """
        lower_output = output.lower()
        if "answer:" in lower_output:
            lower_output = lower_output.split("answer:")[-1].strip()
        return lower_output

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        # Load the test split of the dataset
        for task in tqdm(self.tasks):
            dataset_task = load_dataset(self.huggingface_name, task, split="test", trust_remote_code=True)
            data_list = []
            for sample in dataset_task:
                prompt = self.format_prompt(task, sample)
                answer = sample["answer"].split("####")[-1].strip()
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
            references.append(ref_label)
        return predictions, references

    def GetResult(self, task, prediction, reference):
        metric = evaluate.load("exact_match")
        results = metric.compute(predictions=prediction, references=reference)
        return results

# MBPP: https://huggingface.co/datasets/google-research-datasets/mbpp
# Eval Code Refer to: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/MBPP/eval_instruct.py
class MBPPDataset(DatasetWarperBase):
    name = "MBPP"
    huggingface_name = "google-research-datasets/mbpp"
    tasks = ['full', 'sanitized']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(MBPPDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        将 MBPP 样本信息转化为适合代码生成模型的 Prompt。

        :param task: 任务名称，如 'full'。
        :param sample: 单条数据样本，通常含有 'text', 'code', 'test_list' 等字段。
        :return: str, 拼接完成的 Prompt。
        """
        problem_desc = sample.get('text', '')
        test_list = sample.get('test_list', [])
        tests_str = "\n".join(test_list) if isinstance(test_list, list) else str(test_list)

        prompt = (
            f"Below is a Python problem:\n"
            f"{problem_desc}\n"
            f"To validate your solution, we will run these tests:\n"
            f"{tests_str}\n"
            f"Please write a Python function that solves the above problem.\n"
            # f"{cot_instructions}"
            f"{self.base_template}"  # 如果基类中有模板，这里可直接拼接
        )
        # print(prompt)
        # exit(0)
        return prompt

    def parse_output(self, task, output):
        output = output.split("Answer:")[-1].strip()
        # if output begin with "[" and end with "]", remove them
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        if output.startswith("```python\n"):
            output = output[10:]
        if output.endswith("```"):
            output = output[:-3]
        return output

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        for task in tqdm(self.tasks):
            dataset_task = load_dataset(self.huggingface_name, task, split="test", trust_remote_code=True)
            data_list = []
            for sample in dataset_task:
                prompt = self.format_prompt(task, sample)
                answer = sample["test_list"]
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        # print(f"Total examples: {total_count}")
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
        references = references_text
        return predictions, references

    def GetResult(self, task, prediction, reference):
        """
        对给定的 (prediction, reference) 列表逐条并发执行代码并检查测试用例，
        最后统计通过率并以 {"accuracy": xxx} 的形式返回。

        参数:
            task: 可以根据自身需求扩展，不在本示例中使用。
            prediction: 形如 List[List[str]] 的列表，其中每个子列表包含若干个候选代码字符串。
            reference: 形如 List[str] 的列表，其中每个字符串是该条预测对应的测试用例代码。

        返回:
            {"accuracy": float}，其中 accuracy 可以在此示例中理解为 pass@1。
        """

        # 为了允许在本环境下执行代码测试，需要设置环境变量
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        references_str = [
            "\n".join(test_case_list) for test_case_list in reference
        ]
        prediction_str = [
            [code_str] for code_str in prediction
        ]

        # 这里我们只关心 pass@1，如果需要 pass@10, pass@100 等可自行增加
        k_values = [1]

        # print(prediction_str[0])
        # print(references_str[0])
        # print(prediction_str[1])
        # print(references_str[1])
        # print(prediction_str[2])
        # print(references_str[2])
        # exit(0)
        # 调用 compute_code_eval 函数进行测试
        pass_at_k, results = compute_code_eval(
            predictions=prediction_str,      # 形如 List[List[str]]
            references=references_str,        # 形如 List[str]
            k=k_values,
            num_workers=4,               # 并发线程数，可根据需求调整
            timeout=3.0                  # 单条测试的超时时间，可根据需求调整
        )

        # 提取 pass@1 的得分，如果不存在则默认为 0.0
        accuracy = pass_at_k.get("pass@1", 0.0)

        # 这里可以根据需要打印或记录更详细的结果
        # print("=== Detailed Results ===")
        # for task_id, task_results in results.items():
        #     print(f"Task {task_id}:")
        #     for completion_id, info in task_results:
        #         print(f"  Completion {completion_id}: Passed = {info['passed']}")

        # 返回与原实现相同的结构，只是这里 accuracy 的含义改为 pass@1
        return {"accuracy": accuracy}

# ARC(Easy, Challenge): https://huggingface.co/datasets/allenai/ai2_arc
class ARCDataset(DatasetWarperBase):
    name = "ARC"
    huggingface_name = "allenai/ai2_arc"
    tasks = ['ARC-Easy', 'ARC-Challenge']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(ARCDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, task, sample):
        """
        Formats the prompt for MMLU tasks with optional Chain-of-Thought.

        Args:
            task (str): The specific MMLU task name.
            sample (dict): A dictionary containing:
                - 'question' (str): The question to be answered.
                - 'choices' (List[str]): A list of possible answer choices.

        Returns:
            str: The formatted prompt string.
        """
        # 简要说明
        choices = sample['choices']['label']
        choise_text = ", ".join(choices[:-1]) + " or " + choices[-1]
        # add '
        choises_instruction_text = ", ".join([f"'{choice}'" for choice in choices[:-1]]) + f" or '{choices[-1]}'"
        description = (
            f"The {task} task requires the model to select the most appropriate choice "
            f"({choise_text}) for a given question."
        )
        instructions = (f"Your final answer must be exactly one of: {choises_instruction_text}.")

        # 格式化选项
        choices_formatted = "\n".join([
            f"{sample['choices']['label'][i]}. {choice}" for i, choice in enumerate(sample['choices']['text'])
        ])

        return (
            f"Task: {task}\n"
            f"{description}\n"
            f"Question: {sample['question']}\n"
            f"Choices:\n{choices_formatted}\n"
            f"{instructions}\n"
            f"{self.base_template}"
        )

    def parse_output(self, task, output):
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()
        while output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        try:
            if output[0] in ["A", "B", "C", "D", "E", "1", "2", "3", "4", "5"]:
                return output[0]
        except:
            return -1
        return -1

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        for task in tqdm(self.tasks):
            dataset_task = load_dataset(self.huggingface_name, task, split="test", trust_remote_code=True)
            data_list = []
            for sample in dataset_task:
                # print(sample)
                prompt = self.format_prompt(task, sample)
                answer = sample["answerKey"]
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            # if pred in ["A", "B", "C", "D", "E"], convert to 5, 6, 7, 8, 9
            if pred in ["A", "B", "C", "D", "E"]:
                pred = ord(pred) - ord("A") + 5
            else:
                pred = int(pred)
            predictions.append(pred)
            # Same to reference
            if ref_label in ["A", "B", "C", "D", "E"]:
                ref_label = ord(ref_label) - ord("A") + 5
            else:
                ref_label = int(ref_label)
            references.append(ref_label)
        return predictions, references

    def GetResult(self, task, prediction, reference):
        print(set(reference))
        print(set(prediction))
        metric = evaluate.load("accuracy")
        results = metric.compute(predictions=prediction, references=reference)
        return results

class TableBenchDataset(DatasetWarperBase):
    name = "TableBench"
    huggingface_name = "Multilingual-Multimodal-NLP/TableBench"
    tasks = ['DataAnalysis', 'NumericalReasoning', 'Visualization', 'FactChecking']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        super(TableBenchDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)
        if self.cot_flag:
            self.base_template = (
                "Please first reason step by step, then only provide your final answer after 'Final Answer:'.\n"
                "Please use the following template:\n"
                "Reasoning:[TODO]\n"
                "Final Answer:[TODO]"
            )
        else:
            self.base_template = (
                "Please only provide your final answer after 'Final Answer:'.\n"
                "Please use the following template:\n"
                "Final Answer:[TODO]"
            )

    def format_prompt(self, task, sample): # text, semi, vision
        """
        Formats the prompt for MMLU tasks with optional Chain-of-Thought.

        Args:
            task (str): The specific MMLU task name.
            sample (dict): A dictionary containing:
                - 'question' (str): The question to be answered.
                - 'choices' (List[str]): A list of possible answer choices.

        Returns:
            str: The formatted prompt string.
        """
        # print(sample)
        # print("#" * 20)
        # print(sample['instruction'])
        # print("#" * 20)
        # print(sample['table'])
        # print("#" * 20)
        table_head, table_tail = sample['instruction'].split(f"{sample['table']}")
        # print(table_head)
        table_head = table_head.replace("[TABLE]", "")
        table_head = table_head.replace("\n\n", "\n")
        table_head = table_head.replace("\n\n", "\n")
        table_head = table_head.replace("Read the table below in JSON format:", "")
        table_tail = table_tail.replace("\n\n", "\n")

        if self.mode == "text":
            return sample['instruction']
        elif self.mode == "semi":
            return (
                f"{table_head.strip()}\n"
                f"Read the table in the image\n"
                f"{table_tail.strip()}\n######{sample['table']}\n"
            )
        elif self.mode == "img":
            return (
                f"{table_head.strip()}\n"
                f"Read the table in the below image.\n<img>\n"
                f"{table_tail.strip()}\n######{sample['table']}\n"
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def parse_output(self, task, output):
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()
        # Delete the "[" and "]"
        while output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        return output

    def PureTextData(self):
        data_dict = {}
        total_count = 0
        dataset = load_dataset(self.huggingface_name, split="test", trust_remote_code=True)
        # print(dataset)
        # print(set(dataset["qtype"]))
        # exit(0)
        for task in tqdm(self.tasks):
            # print(len(dataset))
            dataset_task = dataset.filter(lambda x: x["qtype"] == task)
            # print(dataset_task['instruction_type'])
            # print(len(dataset_task))
            if self.cot_flag:
                dataset_task = dataset_task.filter(lambda x: x["instruction_type"] == "SCoT")
            else:
                dataset_task = dataset_task.filter(lambda x: x["instruction_type"] == "DP")
            # print(len(dataset_task))
            # print("#" * 20)
            # exit(0)
            data_list = []
            for sample in dataset_task:
                prompt = self.format_prompt(task, sample)
                answer = sample["answer"]
                data_list.append([prompt, answer])
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    # I rewrite it to make adapt on Table Dataset
    def VisionData(self, vision_type="simple"):
        query_text_template = "Please answer the question in the image."
        DatasetList = self.PureTextData()
        data_dict = self.image_generator.construct_image_dataset(self.name, DatasetList, query_text_template,
                                                                 cot_flag=self.cot_flag, mode=f"table_{self.mode}")
        # exit(0)
        return data_dict

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
            references.append(ref_label)
        return predictions, references

    # TODO
    def GetResult(self, task, prediction, reference, table):
        # metric = evaluate.load("accuracy")
        # results = metric.compute(predictions=prediction, references=reference)
        # return results
        # print(prediction[:10])
        # print(reference[:10])
        if task == "Visualization":
            from TableBench_script import evaluate_visualization
            results = []
            for pred, ref, tab in zip(prediction, reference, table):
                results.append(evaluate_visualization(tab, ref, pred))
            result_dict = {}
            for key in results[0].keys():
                result_dict[key] = sum([res[key] for res in results]) / len(results) * 100
            print(result_dict)
            return result_dict
        else:
            from TableBench_metrics import QAMetric
            metric = QAMetric()
            results = metric.compute(predictions=prediction, references=reference)
            print(results)
            return {"ROUGE-L":results['ROUGE-L']}

    def GetTables(self, task):
        dataset = load_dataset(self.huggingface_name, split="test", trust_remote_code=True)
        table_list = []
        for sample in dataset:
            if sample["qtype"] == task:
                table_list.append(sample['table'])
        # print(f"Total tables: {len(table_list)}")
        # print(table_list[0])
        return table_list

# TODO
class WikiSS_QADataset(DatasetWarperBase):
    name = "WikiSS_QA"
    huggingface_name = "Tevatron/wiki-ss-nq"
    tasks = ['main']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        self.meta_data = None
        super(WikiSS_QADataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    # The same code with SlidesVQA
    def format_prompt(self, question, answer, reference, type): # if type == "img", then reference is a img path

        """
        Formats a prompt based on the type (img or text) and returns the appropriate tuple.

        Args:
            question (str): The question to include in the prompt.
            answer (str): The correct answer to the question.
            reference (str): Reference information (image path or text).
            type (str): Type of reference ("img" or "text").

        Returns:
            tuple: (Prompt, Answer, img_path) for "img" type or (Prompt, Answer) for "text" type.
        """
        if type == "img":
            # Reference as an image
            prompt = (
                f"You are tasked with answering a question based on the provided image. "
                f"For detailed information, refer to the given image."
                f"\n\nQuestion: {question}\n\nAnswer the question based on the image provided."
                f"{self.base_template}"
            )
            return (prompt, reference, answer)
        elif type == "text":
            # Reference as text
            prompt = (
                f"You are tasked with answering a question based on the provided text reference. "
                f"Use the reference to support your answer.\n\nReferences: {reference}\n\n"
                f"Question: {question}\n\nAnswer the question based on the text reference."
                f"{self.base_template}"
            )
            return (prompt, answer)
        else:
            raise ValueError("Invalid type. Type must be 'img' or 'text'.")

    def parse_output(self, task, output):
        output = output.split("Answer:")[-1].strip()
        # if output begin with "[" and end with "]", remove them
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        return output

    def extract_metaData(self):
        import os
        import json
        import random
        from tqdm import tqdm
        from datasets import load_dataset

        cache_dir = "./data/wikiss_subset/"
        metadata_path = os.path.join(cache_dir, "meta_data.jsonl")

        def save_img_to_cache(img, docid):
            """Save a single image to the cache directory."""
            from PIL import Image
            os.makedirs(cache_dir, exist_ok=True)
            img_path = os.path.join(cache_dir, f"{docid}.jpg")
            # Make sure 'img' is a PIL Image. If not, convert it.
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            img.save(img_path, format='JPEG')

        def save_metadata_to_cache(metadata):
            """Save the sampled metadata list to a JSONL file."""
            os.makedirs(cache_dir, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                for entry in metadata:
                    f.write(json.dumps(entry) + "\n")

        # If we already have meta_data, return it
        if getattr(self, "meta_data", None) is not None:
            pass
        elif os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.meta_data = [json.loads(line) for line in f]
        else:
            # Otherwise, we need to generate and cache
            # --- 1) SAMPLE THE DATASET ---
            #    Using reservoir sampling for 3000 items.
            dataset = load_dataset(
                self.huggingface_name,
                split="train",
                trust_remote_code=True,
                streaming=False  # You can do streaming=True if you prefer
            )

            sampled_data = []
            total_count = len(dataset)  # or rely on tqdm's total if known
            for i, data in tqdm(enumerate(dataset), desc="Sampling dataset", total=total_count):
                if len(sampled_data) < 3000:
                    sampled_data.append(data)
                else:
                    r = random.random()
                    if r < 3000 / (i + 1):
                        replace_idx = random.randint(0, 2999)
                        sampled_data[replace_idx] = data

            print(f"Number of sampled items: {len(sampled_data)}")

            # --- 2) GATHER RELEVANT DOCIDS ---
            needed_docids = set()
            for data in sampled_data:
                positive_passages = data.get("positive_passages", [])
                for passage in positive_passages:
                    if "docid" in passage:
                        needed_docids.add(passage["docid"])

            # --- 3) STREAM THROUGH THE CORPUS AND SAVE ONLY THE NEEDED IMAGES ---
            # Instead of loading the entire dataset_corpus into memory, we iterate over it just once.
            dataset_corpus = load_dataset(
                "Tevatron/wiki-ss-corpus",
                split="train",
                trust_remote_code=True,
                streaming=True  # crucial to avoid OOM with very large dataset
            )

            # For each item, if its docid is in needed_docids, save the image.
            for entry in tqdm(dataset_corpus, desc="Saving images for sampled docids"):
                docid = entry["docid"]
                if docid in needed_docids:
                    image = entry["image"]
                    save_img_to_cache(image, docid)
                    # Once saved, we can remove it from needed_docids to skip duplicates
                    needed_docids.remove(docid)
                    # Optional early stop: if we've saved all needed docids, we can break
                    if not needed_docids:
                        break

            # --- 4) SAVE THE METADATA AND RETURN ---
            self.meta_data = sampled_data
            save_metadata_to_cache(self.meta_data)
        # return self.meta_data

        # Random select 5000 datapoint from dataset, and for all len(positive_passages)==1 datapoint, save their img
        # After that, save the datapoint_meta_data to cache_dir/meta_data.jsonl
        # print(self.meta_data[0])
        print(len(self.meta_data[0]['positive_passages']))
        meta_data = []
        for i in range(len(self.meta_data)):
            question = self.meta_data[i]['query']
            answer = ",".join(self.meta_data[i]['answers'])
            text_inputs, img_inputs = [], []
            for passage in self.meta_data[i]['positive_passages']:
                img = "./data/wikiss_subset/" + passage['docid'] + ".jpg"
                text = passage['text']
                text_inputs.append(text)
                img_inputs.append(img)
            meta_data.append([question, answer, text_inputs, img_inputs])
        print(len(meta_data))
        return meta_data

    def PureTextData(self, type = "text"):
        data_dict = {}
        total_count = 0
        meta_data = self.extract_metaData()
        for task in tqdm(self.tasks):
            data_list = []
            for data_point in meta_data:
                if type == "text":
                    data_list.append(self.format_prompt(data_point[0], data_point[1], data_point[2], type = type))
                elif type == "img":
                    data_list.append(self.format_prompt(data_point[0], data_point[1], data_point[3], type = type))
                else:
                    raise ValueError(f"Invalid type: {type}")
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def VisionData(self, vision_type="simple"):
        return self.PureTextData(type="img")

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
            references.append(ref_label)
        return predictions, references

    def GetResult(self, task, prediction, reference):
        from TableBench_metrics import QAMetric
        metric = QAMetric()
        results = metric.compute(predictions=prediction, references=reference)
        print(results)
        return {"ROUGE-L": results['ROUGE-L']}

class SlidesVQADataset(DatasetWarperBase):
    name = "SlidesVQA"
    huggingface_name = ""
    tasks = ['main']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        self.meta_data = None
        super(SlidesVQADataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def format_prompt(self, question, answer, references, type): # if type == "img", then reference is a img path

        """
        Formats a prompt based on the type (img or text) and returns the appropriate tuple.

        Args:
            question (str): The question to include in the prompt.
            answer (str): The correct answer to the question.
            reference (str): Reference information (image path or text).
            # Update 0129: references is a list of text or img path
            type (str): Type of reference ("img" or "text").

        Returns:
            tuple: (Prompt, Answer, img_path) for "img" type or (Prompt, Answer) for "text" type.
        """
        if type == "img":
            # Reference as an image
            prompt = (
                f"You are tasked with answering a question based on the provided images. "
                f"For detailed information, refer to the given image."
                f"\n\nQuestion: {question}\n\nAnswer the question based on the image provided."
                f"{self.base_template}"
            )
            return (prompt, references, answer)
        elif type == "text":
            # Reference as text
            references_text = ""
            for i in range(len(references)):
                references_text += f"{i}. {references[i]}\n"
            prompt = (
                f"You are tasked with answering a question based on the provided text reference. "
                f"Use the reference to support your answer.\n\nReferences: \n{references_text}\n\n"
                f"Question: {question}\n\nAnswer the question based on the text reference."
                f"{self.base_template}"
            )
            return (prompt, answer)
        else:
            raise ValueError("Invalid type. Type must be 'img' or 'text'.")

    def parse_output(self, task, output):
        output = output.split("Answer:")[-1].strip()
        # if output begin with "[" and end with "]", remove them
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        return output

    def extract_metaData(self):
        if not (self.meta_data is None):
            return self.meta_data
        from SlidesVQA_script import extract_question_answer_pairs, extract_query_positive_pairs
        qa_pairs, qa_duplicates = extract_question_answer_pairs(
            "./data/annotations/qa/test.jsonl")
        query_positive_pairs, query_duplicates = extract_query_positive_pairs(
            "./data/slide-data/test-new.jsonl")
        # print(len(qa_pairs), len(query_positive_pairs))
        dict_QA = {}
        dict_QP = {}
        for i in range(len(qa_pairs)):
            dict_QA[qa_pairs[i][0]] = qa_pairs[i][1]
        for i in range(len(query_positive_pairs)):
            dict_QP[query_positive_pairs[i]['query']] = query_positive_pairs[i]['positive_passages']
        meta_data = []
        for key in dict_QP:
            question = key
            answer = dict_QA[key]
            pe = dict_QP[key]
            text_inputs, img_inputs = [], []
            for i in range(len(pe)):
                text_inputs.append(pe[i]['text'])
                img_inputs.append(f"./data/slide-data/all_slides/{pe[i]['docid']}.jpg")
            # text_input = pe['text']
            # img_input = f"./data/slide-data/all_slides/{pe['docid']}.jpg"
            meta_data.append([question, answer, text_inputs, img_inputs])
        self.meta_data = meta_data
        print("metadata:", len(meta_data))
        # xit(0)
        return meta_data

    def PureTextData(self, type = "text"):
        data_dict = {}
        total_count = 0
        meta_data = self.extract_metaData()
        for task in tqdm(self.tasks):
            data_list = []
            for data_point in meta_data:
                if type == "text":
                    data_list.append(self.format_prompt(data_point[0], data_point[1], data_point[2], type = type))
                elif type == "img":
                    data_list.append(self.format_prompt(data_point[0], data_point[1], data_point[3], type = type))
                else:
                    raise ValueError(f"Invalid type: {type}")
                data_dict[task] = data_list
                total_count += 1
        print(f"Total examples: {total_count}")
        return data_dict

    def VisionData(self, vision_type="simple"):
        return self.PureTextData(type="img")

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            predictions.append(pred)
            references.append(ref_label)
        return predictions, references

    def GetResult(self, task, prediction, reference):
        from TableBench_metrics import QAMetric
        metric = QAMetric()
        results = metric.compute(predictions=prediction, references=reference)
        print(results)
        return {"ROUGE-L": results['ROUGE-L']}

class MathverseDataset(DatasetWarperBase):
    name = "Mathverse"
    huggingface_name = "AI4Math/MathVerse"
    tasks = ['main']
    image_generator = ImageGenerator()

    def __init__(self, cot_flag=True, mode='text', init_data = True, from_hf = False):
        self.meta_data = None
        super(MathverseDataset, self).__init__(cot_flag=cot_flag, mode=mode, init_data=init_data, from_hf=from_hf)

    def parse_output(self, task, output):
        output = output.split("Answer:")[-1].strip()
        # if output begin with "[" and end with "]", remove them
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        return output

    def extract_metaData(self, subset = "text"): # text, img
        # if subset = "text", return list of (text, answer), otherwise, return (text, img_path, answer)
        import os
        def calc_path(path):
            return os.path.join("./data/mathverse/images", path)
        if subset == "text":
            path = "./data/mathverse/testmini_text_only.json"
            problem_version = "Text Only"

        else:
            path = "./data/mathverse/testmini.json"
            problem_version = "Vision Only"

        try:
            with open(path, "r") as file:
                data = json.load(file)
            extracted_data = []
            for item in data:
                if item["problem_version"] == problem_version:
                    text = item["query_cot"]
                    answer = item["answer"]

                    if subset == "text":
                        extracted_data.append((text, answer))
                    else:
                        img_path = calc_path(item["image"])
                        extracted_data.append((text, img_path, answer))

            return extracted_data

        except FileNotFoundError:
            print(f"File not found: {path}")
            return []
        except json.JSONDecodeError:
            print("Error decoding JSON file.")
            return []

    def PureTextData(self, type = "text"):
        data_dict = {}
        total_count = 0
        meta_data = self.extract_metaData(subset = type)
        # print(len(meta_data))
        # exit(0)
        for task in tqdm(self.tasks):
            data_list = meta_data
            data_dict[task] = data_list
        print(f"Total examples: {total_count}")
        return data_dict

    def VisionData(self, vision_type="simple"):
        return self.PureTextData(type="img")

    def FormatPredNRef(self, dataset, task, predictions_text, references_text, eval_time = True):
        predictions, references = [], []
        for idx, (p_out, ref_label) in enumerate(zip(predictions_text, references_text)):
            pred = self.parse_output(task, p_out)
            ref = self.parse_output(task, ref_label)
            predictions.append(pred)
            references.append(ref)
        return predictions, references

    def parse_output(self, task, output):
        """
        Parses the model output for MMLU tasks.
        1) 提取并仅保留 'Answer:' 后面的内容。
        2) 将结果转成 a, b, c, d 等小写后映射为索引。

        Args:
            task (str): The specific MMLU task name.
            output (str): The raw output string from the model.

        Returns:
            int: The index of the chosen option (0 for A, 1 for B, etc.), or -1 if invalid.
        """
        # print(output)
        lower_output = output.lower()
        if "answer:" in lower_output:
            lower_output = lower_output.split("answer:")[-1].strip()
        lower_output = lower_output.strip()
        if lower_output in ["a", "b", "c", "d"]:
            return ord(lower_output) - ord("a")
        return -1

    def GetResult(self, task, prediction, reference):
        # prediction = [self.parse_output(task, p) for p in prediction]
        # reference = [self.parse_output(task, r) for r in reference]
        metric = evaluate.load("accuracy")
        results = metric.compute(predictions=prediction, references=reference)
        print(results)
        return {"accuracy": results['accuracy']}

        # from TableBench_metrics import QAMetric
        # metric = QAMetric()
        # results = metric.compute(predictions=prediction, references=reference)
        # print(results)
        # return {"ROUGE-L": results['ROUGE-L']}

def main():
    parser = argparse.ArgumentParser(description="Run SuperGLUE inference and evaluation.")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="The dataset class name to use, e.g., 'SuperGLUEDataset'."
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="The model wrapper class name to use, e.g., 'QWen2VLWrapper'."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['text', 'img', 'semi'],
        help="The mode for inference and evaluation. Options are 'text', 'img' or 'semi'."
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        choices=['base', 'cot'],
        help="The prompt type for text data. Options are 'base' or 'cot'."
    )

    parser.add_argument(
        '--from_hf',
        action='store_true',
        help="Using Local or Huggingface dataset."
    )

    args = parser.parse_args()

    from model import load_modelObject
    model = load_modelObject(args.model, input_method = args.mode)

    # Access dataset class from current file
    dataset_class = globals().get(args.dataset)
    if dataset_class is None:
        raise ValueError(f"Dataset class '{args.dataset}' not found in the current file.")

    # Initialize dataset and model instances
    dataset = dataset_class(cot_flag= args.prompt == 'cot', mode=args.mode, from_hf = args.from_hf)
    if args.mode == 'img':
        dataset.VisionData()
        print("Image data generated successfully.")
    if args.mode == 'semi':
        dataset.VisionData(vision_type="semi")
        print("Semi-Vision data generated successfully.")

    # Set the model cache path
    if "GPT" in args.model:
        model.change_cache_path(f"cache_gpt4o/{args.dataset}_text.jsonl",
                                f"cache_gpt4o/{args.dataset}_img.jsonl",
                                f"cache_gpt4o/{args.dataset}_token_usage.jsonl")
    elif "Gemini" in args.model:
        model.change_cache_path(f"cache_geminiflash/{args.dataset}_text.jsonl",
                                f"cache_geminiflash/{args.dataset}_img.jsonl",
                                f"cache_geminiflash/{args.dataset}_token_usage.jsonl")

    # Run the inference with a timer
    start_time = time.time()
    dataset.InferenceDataset(model)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total inference time: {total_time:.2f} seconds")

    # TODO: Run the evaluation
    dataset.Eval(model.name)

def testDataset(dataset_name = "TableBenchDataset"):
    # print("Test dataset: ", dataset_name)
    # exit(0)
    dataset_class = globals().get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Dataset class '{dataset_name}' not found in the current file.")
    from model import TestWrapper
    model = TestWrapper()

    dataset = dataset_class(cot_flag=True, mode='text', from_hf = True) # test text mode
    dataset.InferenceDataset(model)
    dataset.Eval(model.name)

    dataset = dataset_class(cot_flag=True, mode='img', from_hf = True) # test text mode

    # Run the inference with a timer
    start_time = time.time()
    dataset.InferenceDataset(model)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total inference time: {total_time:.2f} seconds")

    dataset.Eval(model.name)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        testDataset("ARCDataset")
    # main()
    # testDataset("WikiSS_QADataset")
    # testDataset("SlidesVQADataset")
    # testDataset("MMLUProDataset")
    # testDataset("ARCDataset")

