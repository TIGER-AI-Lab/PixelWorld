# This code is from https://github.com/TIGER-AI-Lab/MEGA-Bench/tree/a85afe936efab2ec498886e20967b41a1b25db91/megabench/models
import base64
import logging
from PIL import Image, ImageFile
from io import BytesIO
from tqdm import tqdm
import pathlib
import json
import re
import abc
import torch
import warnings
import math
from typing import List, Dict, Union
from mimetypes import guess_type
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from anthropic.types import message
from anthropic import (
    InternalServerError,
    BadRequestError,
    RateLimitError,
    APIStatusError
)

# import llava
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from llava.conversation import conv_templates

# Constants
MAX_OUTPUT_TOKENS = 8192
MAX_TOKENS_PER_IMAGE = 4096
SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("infoLogger")

# PIL Image Configuration
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_video_file(file_path):
    mime_type, _ = guess_type(file_path)
    if not mime_type:
        return False
    return mime_type.startswith("video")

class BaseModel(abc.ABC):
    def __init__(
            self,
            api_key=None,
            model=None,
            query_data=None,
            resize=True,
            max_side=1000,
            print_response=False,
            max_num_image=1,
            system_message: Union[str, None] = None,
            total_demo_video_frames=4,
            **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.query_data = query_data
        self.resize = resize
        self.max_side = max_side
        self.print_response = print_response
        self.prompts = self.load_prompts(
            pathlib.Path(__file__).resolve().parent / "prompt.json"
        )
        # the maximum number of images in each model API query
        self.max_num_image = max_num_image
        self.system_message = system_message
        self.total_demo_video_frames = total_demo_video_frames  # the number of frames sampled for videos for the demonstration examples
        # set number of max demo image to be the same as the number of frames per demo video
        self.max_demo_num_image = total_demo_video_frames

        # To be determined per task
        self.query_video_frames = None
        self.demo_video_frames = None

    @staticmethod
    def load_prompts(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_system_message(self) -> List[Dict[str, str]]:
        if self.system_message:
            return [{"role": "system", "content": self.system_message}]
        else:
            return []

    @abc.abstractmethod
    def prepare_context(self):
        pass

    @abc.abstractmethod
    def prepare_example_content(self, example_info):
        pass

    @abc.abstractmethod
    def prepare_query_content(self, query_info):
        pass

    @abc.abstractmethod
    def create_image_content(self, image_path):
        pass

    @staticmethod
    def _is_video_file(file_path):
        return is_video_file(file_path)

    def _set_sampling_config(self, query_idx):
        query_data = self.query_data
        num_query_videos = 0
        num_query_images = 0
        num_demo_videos = 0
        num_demo_images = 0
        for global_img in query_data["global_images"]:
            if self._is_video_file(global_img):
                num_query_videos += 1
            else:
                num_query_images += 1

        demo_example = query_data["example_info"]
        for demo_img in demo_example["image_paths"]:
            if self._is_video_file(demo_img):
                num_demo_videos += 1
            else:
                num_demo_images += 1

        query_example = query_data["queries"][query_idx]
        for query_img in query_example["image_paths"]:
            if self._is_video_file(query_img):
                num_query_videos += 1
            else:
                num_query_images += 1

        # the actual number of demo images to be used
        num_demo_images = min(self.max_demo_num_image, num_demo_images)

        if self.max_num_image:
            if num_demo_videos > 0:
                self.demo_video_frames = math.ceil(
                    self.total_demo_video_frames / num_demo_videos
                )
            else:
                self.demo_video_frames = 0

            if num_query_videos > 0:
                total_query_video_frames = (
                        self.max_num_image
                        - num_demo_images
                        - num_query_images
                        - self.demo_video_frames * num_demo_videos
                )
                if total_query_video_frames <= 0:
                    raise ValueError(
                        f"Cannot query <= 0 frames: please raise the number of maximum images allowed. {self.demo_video_frames=} {num_demo_videos=} {self.max_num_image=}"
                    )
                self.query_video_frames = total_query_video_frames // num_query_videos
            else:
                self.query_video_frames = 0

            total_num_image = (
                    self.query_video_frames * num_query_videos
                    + self.demo_video_frames * num_demo_videos
                    + num_query_images
                    + num_demo_images
            )
            exceed_image_quota = total_num_image > self.max_num_image
        else:
            exceed_image_quota = False

        return exceed_image_quota

    def process_video(self, video_path, is_demo):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        # assert (
        #     "video_sampling" in self.query_data
        # ), "Missing video sampling strategy setting..."
        num_frames = self.demo_video_frames if is_demo else self.query_video_frames

        # the sampling rate using max number of frames
        sampling_gap_maxframe = (
            1 if not num_frames else math.ceil(frame_count / num_frames)
        )

        # If not set up, determine the sampling based on the video fps
        video_sampling = self.query_data.get("video_sampling", "fps")

        if video_sampling == "max":
            if fps >= 10:
                sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)
            else:
                sampling_gap = sampling_gap_maxframe
        elif video_sampling == "fps":
            sampling_gap_fps = (
                math.ceil(frame_count / self.demo_video_frames)
                if is_demo
                else math.ceil(fps)
            )
            sampling_gap = max(sampling_gap_fps, sampling_gap_maxframe)

        frame_number = 0
        images = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            # Sample frames based on the dynamic sampling rate
            if frame_number % sampling_gap == 0:
                # Create a temporary file for the frame
                with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                ) as temp_frame:
                    cv2.imwrite(temp_frame.name, frame)
                    images.append(self.create_image_content(temp_frame.name))
                    os.remove(temp_frame.name)
            frame_number += 1
        if frame_number == 0:
            raise ValueError(f"Failed to read video from {video_path}, check data...")
        logging.info(
            f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
        )
        cap.release()
        return images

    def create_media_content(self, file_path, is_demo=False):
        pass

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def _resize_image(self, image):
        resize_scale = self.max_side / max(image.size)
        new_size = (
            int(image.size[0] * resize_scale),
            int(image.size[1] * resize_scale),
        )
        return image.resize(new_size)

    @abc.abstractmethod
    def query(self, task_name, query_data, position=0):
        pass

    def clear(self):
        self.api_key = None
        self.model = None
        self.query_data = None
        self.resize = None
        self.max_side = None
        self.prompts = None

    def _process_text_and_media(self, text, media_paths, is_example=False):
        content = []
        chunks = re.split(r'(<image>|<video>)', text)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])

        if placeholder_count != len(media_paths):
            raise ValueError(
                f"Mismatch between number of placeholders ({placeholder_count}) and media paths ({len(media_paths)})")

        media_index = 0
        curr_demo_images = 0
        for chunk in chunks:
            if chunk in ['<image>', '<video>']:
                media_content = self.create_media_content(media_paths[media_index], is_demo=is_example)
                if len(media_content) == 1:  # image
                    if is_example and curr_demo_images >= self.max_demo_num_image:
                        logging.warning("Exceed the quota for demo image, skip the demo image")
                    else:
                        content.extend(media_content)
                        if is_example:
                            curr_demo_images += 1
                else:  # video
                    content.extend(media_content)
                media_index += 1
            elif chunk.strip():  # Only add non-empty text chunks
                content.append({"type": "text", "text": chunk.strip()})

        return content

class Claude(BaseModel):

    ATTEMPT_LIMIT = 5

    def create_media_content(self, file_path, is_demo=False):
        if self._is_video_file(file_path):
            # Handle video processing with the frame subsampling logic
            video_content = [{"type": "text", "text": self.prompts["video_start"]}]
            video_content.extend(self.process_video(file_path, is_demo))
            video_content.append({"type": "text", "text": self.prompts["video_end"]})
            return video_content
        else:
            # Handle image processing otherwise
            return [self.create_image_content(file_path)]

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        return {
            "type": "image",
            "source": {"media_type": mime_type, "type": "base64", "data": base64_image},
        }

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

        image = Image.open(image_path)
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)
        if self.resize and max(image.size) > self.max_side:
            image = self._resize_image(image)
        encoded_image = self._encode_image(image, image_format)

        return encoded_image, mime_type

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    def prepare_context(self):
        global_description = self.query_data.get("global_description", "")
        global_images = self.query_data.get("global_images", [])
        content = self._process_text_and_media(global_description, global_images)

        example_info = self.query_data["example_info"]
        example_content = self.prepare_example_content(example_info)
        content.extend(example_content)
        return content

    def prepare_example_content(self, example_info):
        example_text = example_info["example_text"]
        example_media_paths = example_info["image_paths"]
        return self._process_text_and_media(
            example_text, example_media_paths, is_example=True
        )

    def prepare_query_content(self, query_info):
        query_text = query_info.get("query_text", "")
        image_paths = query_info.get("image_paths", [])
        query_content = self._process_text_and_media(query_text, image_paths)
        return query_content

    def make_client(self):
        return anthropic.Anthropic(api_key=self.api_key, max_retries=5)

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        client = self.make_client()

        self._set_sampling_config(0)
        context = self.prepare_context()

        query_response = []
        for query_idx, query_info in enumerate(
            tqdm(
                query_data["queries"],
                desc=f"{task_name} - Processing queries",
                unit="query",
                position=position,
            )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)
            if not exceed_image_quota:
                query_content = self.prepare_query_content(query_info)

                messages = self.prepare_system_message()
                messages.append({"role": "user", "content": context + query_content})

                response = None
                n_attempt = 0
                while response is None:
                    try:
                        response: anthropic.types.Message = client.messages.create(
                            model=self.model,
                            temperature=0,
                            max_tokens=4096,
                            messages=messages,
                        )
                        time.sleep(1)
                    except (BadRequestError, InternalServerError) as e:
                        n_attempt += 1
                        logging.info(
                            f"Got error: {e}. Retry... Attemp times: {n_attempt}"
                        )
                        time.sleep(2)
                        if n_attempt >= self.ATTEMPT_LIMIT:
                            response = str(e)
                    except RateLimitError as e:
                        n_attempt += 1
                        logging.info(
                            f"Got RateLimit error: {e}. Retry... Attemp times: {n_attempt}"
                        )
                        time.sleep(2**n_attempt)
                    except APIStatusError as e:
                        response = str(e)

                if isinstance(response, message.Message):
                    text_content = response.content[0].text
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                else:
                    text_content = response
                    input_tokens = output_tokens = "NA"

                if self.print_response:
                    logging.info(
                        f"response: {text_content} ; tokens: {input_tokens + output_tokens}"
                    )

            else:
                text_content = (
                    "Exceed the specified max number of images, skip running the model."
                )
                logging.info(text_content)

            query_response.append(
                {
                    "response": text_content,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response

class OpenAI(BaseModel):

    def create_media_content(self, file_path, is_demo=False):
        if self._is_video_file(file_path):
            # Handle video processing with the frame subsampling logic
            video_content = [{"type": "text", "text": self.prompts["video_start"]}]
            video_content.extend(self.process_video(file_path, is_demo))
            video_content.append({"type": "text", "text": self.prompts["video_end"]})
            return video_content
        else:
            # Handle image processing otherwise
            return [self.create_image_content(file_path)]

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    def encode_image(self, image_path, max_side=None):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)
        if not max_side and self.max_side:
            max_side = self.max_side

        if self.resize and max(image.size) > self.max_side:
            image = self._resize_image(image)
        encoded_image = self._encode_image(image, image_format)

        return encoded_image, mime_type

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    def prepare_context(self):
        global_description = self.query_data.get("global_description", "")
        global_images = self.query_data.get("global_images", [])
        content = self._process_text_and_media(global_description, global_images)

        example_info = self.query_data["example_info"]
        example_content = self.prepare_example_content(example_info)
        content.extend(example_content)
        return content

    def prepare_example_content(self, example_info):
        example_text = example_info["example_text"]
        example_media_paths = example_info["image_paths"]
        return self._process_text_and_media(example_text, example_media_paths, is_example=True)

    def prepare_query_content(self, query_info):
        query_text = query_info.get("query_text", "")
        image_paths = query_info.get("image_paths", [])
        query_content = self._process_text_and_media(query_text, image_paths)
        return query_content

    @property
    def url(self) -> str:
        """The server URL."""
        return self._url if hasattr(self, '_url') else "https://api.openai.com/v1/chat/completions"

    @url.setter
    def url(self, value: str) -> None:
        """Set the server URL."""
        self._url = value

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self._set_sampling_config(0)
        context = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
            tqdm(
                query_data["queries"],
                desc=f"{task_name} - Processing queries",
                unit="query",
                position=position,
            )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)

            query_content = self.prepare_query_content(query_info)

            if not exceed_image_quota:
                messages = self.prepare_system_message()
                messages.append({"role": "user", "content": context + query_content})
                query_payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                }

                response_data = None
                while response_data is None:
                    response = requests.post(
                        self.url,
                        headers=headers,
                        json=query_payload,
                    )
                    try:
                        response_ = response.json()
                    except requests.exceptions.JSONDecodeError as e:
                        logging.info(f"Can't parse output: {e}, retry...")
                    if "error" in response_:
                        error_info = response_["error"]
                        logging.info(
                            f"Got error with type: {error_info['type']}. Message: {error_info['message']}"
                        )
                        logging.info(f"Retry...")
                    else:
                        response_data = response_
                        break

                total_tokens = response_data.get("usage", {}).get("total_tokens", "N/A")

                # Extracting the 'content' field from the response
                if response_data and "choices" in response_data:
                    choices = response_data["choices"]
                    if choices and "message" in choices[0]:
                        message_content = choices[0]["message"]["content"]
                        if self.print_response:
                            logging.info(
                                f"response: {message_content}; tokens:{total_tokens}"
                            )
                else:
                    message_content = ""
            else:
                message_content = (
                    "Exceed the specified max number of images, skip..."
                )

            # save the correct answer here as well for later scoring
            query_response.append(
                {
                    "response": message_content,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response

class InternVL(OpenAI):
    def __init__(
            self,
            api_key=None,
            model=None,
            query_data=None,
            resize=True,
            max_side=1000,
            print_response=False,
            max_num_image=1,
            system_message=None,
            total_demo_video_frames=4,
            **kwargs,
    ):
        super().__init__(
            api_key,
            model,
            query_data,
            resize,
            max_side,
            print_response,
            max_num_image,
            system_message,
            total_demo_video_frames,
            **kwargs,
        )

        # load config from config.json
        self.config = self.load_config(
            pathlib.Path(__file__).resolve().parent / "config.json"
        )
        if model is None:
            raise ValueError("Model name is required for InternVL")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        self.llm = LLM(
            model=model,
            max_num_seqs=8,
            max_model_len=8192,
            limit_mm_per_prompt={"image": max_num_image},
            trust_remote_code=True,
            tensor_parallel_size=kwargs.get("ngpus", 1),
            gpu_memory_utilization=kwargs.get("gpu_utils", 0.9)
        )

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        image = Image.open(BytesIO(base64.b64decode(base64_image)))
        image.load()
        return image

    def create_media_content(self, file_path, is_demo=False):
        if self._is_video_file(file_path):
            # Handle video processing with the frame subsampling logic
            return self.process_video(file_path, is_demo)
        else:
            # Handle image processing otherwise
            return [self.create_image_content(file_path)]

    def prepare_context(self):
        content, images = [], []
        global_description = self.query_data.get("global_description", "")
        global_image_paths = self.query_data.get("global_images", [])

        context_content, context_images = self._process_text_and_media(global_description, global_image_paths)
        content.extend(context_content)
        images.extend(context_images)

        example_info = self.query_data["example_info"]
        example_content, example_images = self.prepare_example_content(example_info)
        content.extend(example_content)
        images.extend(example_images)

        return content, images

    def prepare_example_content(self, example_info):
        example_text = example_info["example_text"]
        example_media_paths = example_info["image_paths"]
        example_content, example_images = self._process_text_and_media(example_text, example_media_paths,
                                                                       is_example=True)

        return example_content, example_images

    def prepare_query_content(self, query_info):
        query_text = query_info.get("query_text", "")
        image_paths = query_info.get("image_paths", [])

        query_content, query_images = self._process_text_and_media(query_text, image_paths)


        return query_content, query_images

    def _set_patch_strategy(self, images, use_one=False):
        if len(images) > 8 or use_one:
            self.llm.llm_engine.model_config.hf_config.use_thumbnail = False
            self.llm.llm_engine.model_config.hf_config.max_dynamic_patch = 1
        elif len(images) > 4:
            self.llm.llm_engine.model_config.hf_config.use_thumbnail = True
            self.llm.llm_engine.model_config.hf_config.max_dynamic_patch = 2
        elif len(images) > 2:
            self.llm.llm_engine.model_config.hf_config.use_thumbnail = True
            self.llm.llm_engine.model_config.hf_config.max_dynamic_patch = 4
        else:
            self.llm.llm_engine.model_config.hf_config.use_thumbnail = True
            self.llm.llm_engine.model_config.hf_config.max_dynamic_patch = 6

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        self._set_sampling_config(0)
        context, context_image_list = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
                tqdm(
                    query_data["queries"],
                    desc=f"{task_name} - Processing queries",
                    unit="query",
                    position=position,
                )
        ):
            generated_text = None
            exceed_image_quota = self._set_sampling_config(query_idx)
            if not exceed_image_quota:
                images = []
                query_content, query_images = self.prepare_query_content(query_info)
                if context_image_list:
                    images.extend(context_image_list)
                if query_images:
                    images.extend(query_images)

                query_payload_list = context + query_content

                # Do not update the image patching strategy if in single-image setting
                # (all tasks will use only one image)
                self._set_patch_strategy(images)
                query_payload = "\n".join(query_payload_list)
                message = [{"role": "user", "content": query_payload}]

                prompt = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
                stop_token_ids = [
                    self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens
                ]
                max_new_tokens = self.config.get("session_len", 8192)
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    stop_token_ids=stop_token_ids,
                )

                try:
                    outputs = self.llm.generate(
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": images},
                        },
                        sampling_params=sampling_params,
                    )
                except (ValueError, RuntimeError):
                    # the prompt is too long, or the image patching leads to different image size, 
                    # Then force to reset the processor, fall back to use one image
                    self._set_patch_strategy(images, use_one=True)
                    try:
                        outputs = self.llm.generate(
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": images},
                            },
                            sampling_params=sampling_params,
                        )
                    except Exception as e:
                        # if still fail, output the error info
                        outputs = str(e)

                if isinstance(outputs, list):
                    for o in outputs:
                        generated_text = o.outputs[0].text
                else:
                    generated_text = outputs

            else:
                generated_text = (
                    "Exceed the specified max number of images, skip running the model."
                )

            if self.print_response:
                print(f"Model response:\n {generated_text}")

            query_response.append(
                {
                    "response": generated_text,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response

    def _process_text_and_media(self, text, media_paths, is_example=False):
        # print(text, media_paths)
        # print("#"*100)
        content = []
        images = []
        chunks = re.split(r'(<image>|<video>)', text)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])

        if placeholder_count != len(media_paths):
            raise ValueError(
                f"Mismatch between number of placeholders ({placeholder_count}) and media paths ({len(media_paths)})")

        media_index = 0
        curr_demo_images = 0
        for chunk in chunks:
            if chunk == '<image>':
                if is_example and curr_demo_images >= self.max_demo_num_image:
                    logging.warning("Exceed the quota for demo image, skip the demo image")
                else:
                    image_content = self.create_media_content(media_paths[media_index], is_demo=is_example)
                    content.append("<image>")
                    images.extend(image_content)
                    if is_example:
                        curr_demo_images += 1
                media_index += 1
            elif chunk == '<video>':
                video_content = self.create_media_content(media_paths[media_index], is_demo=is_example)
                content.append(self.prompts["video_start"])
                for _ in video_content:
                    content.append("<image>")
                content.append(self.prompts["video_end"])
                images.extend(video_content)
                media_index += 1
            elif chunk.strip():  # Only add non-empty text chunks
                content.append(chunk.strip())

        return content, images

# class LlavaOV(OpenAI):
#     def __init__(
#         self,
#         api_key=None,
#         model=None,
#         query_data=None,
#         resize=True,
#         max_side=1000,
#         print_response=False,
#         max_num_image=1,
#         system_message=None,
#         total_demo_video_frames=4,
#         **kwargs,
#     ):
#         super().__init__(
#             api_key,
#             model,
#             query_data,
#             resize,
#             max_side,
#             print_response,
#             max_num_image,
#             system_message,
#             total_demo_video_frames,
#             **kwargs,
#         )
#
#         # load config from config.json
#         self.config = self.load_config(
#             pathlib.Path(__file__).resolve().parent / "config.json"
#         )
#
#         llava_model_args = {
#             "multimodal": True,
#         }
#         overwrite_config = {}
#         overwrite_config["image_aspect_ratio"] = "pad"
#         llava_model_args["overwrite_config"] = overwrite_config
#         self.tokenizer, self.model, self.image_processor, self.max_length = (
#             load_pretrained_model(
#                 model, None, "llava_qwen", device_map="auto", **llava_model_args
#             )
#         )
#
#         self.model.eval()
#
#     @staticmethod
#     def load_config(file_path):
#         with open(file_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#
#     def create_image_content(self, image_path):
#         image, mime_type = self.encode_image(image_path)
#         return image
#
#     def encode_image(self, image_path, max_side=None):
#         mime_type, _ = guess_type(image_path)
#         if mime_type is None:
#             mime_type = "image/jpeg"
#
#         image = Image.open(image_path)
#         # Handle the alpha channel
#         if image.mode == "RGBA":
#             image = self._rgba_to_rgb(image)
#
#         # if not specified, using the
#         if not max_side and self.max_side:
#             max_side = self.max_side
#
#         if self.resize and max(image.size) > self.max_side:
#             image = self._resize_image(image)
#
#         return image.convert("RGB"), mime_type
#
#     def create_media_content(self, file_path, is_demo=False):
#         if self._is_video_file(file_path):
#             # Handle video processing with the frame subsampling logic
#             return self.process_video(file_path, is_demo)
#         else:
#             # Handle image processing otherwise
#             return [self.create_image_content(file_path)]
#
#     def _process_text_and_media(self, text, media_paths, is_example=False):
#         content = []
#         images = []
#         chunks = re.split(r"(<image>|<video>)", text)
#
#         placeholder_count = sum(
#             1 for chunk in chunks if chunk in ["<image>", "<video>"]
#         )
#
#         if placeholder_count != len(media_paths):
#             raise ValueError(
#                 f"Mismatch between number of placeholders ({placeholder_count}) and media paths ({len(media_paths)})"
#             )
#
#         media_index = 0
#         curr_demo_images = 0
#         for chunk in chunks:
#             if chunk == "<image>":
#                 if is_example and curr_demo_images >= self.max_demo_num_image:
#                     logging.warning(
#                         "Exceed the quota for demo image, skip the demo image"
#                     )
#                 else:
#                     image_content = self.create_media_content(
#                         media_paths[media_index], is_demo=is_example
#                     )
#                     content.append(DEFAULT_IMAGE_TOKEN)
#                     images.extend(image_content)
#                     if is_example:
#                         curr_demo_images += 1
#                 media_index += 1
#             elif chunk == "<video>":
#                 video_content = self.create_media_content(
#                     media_paths[media_index], is_demo=is_example
#                 )
#                 content.append(self.prompts["video_start"])
#                 for _ in video_content:
#                     content.append(DEFAULT_IMAGE_TOKEN)
#                 content.append(self.prompts["video_end"])
#                 images.extend(video_content)
#                 media_index += 1
#             elif chunk.strip():  # Only add non-empty text chunks
#                 content.append(chunk.strip())
#
#         return content, images
#
#     def prepare_context(self):
#         content, images = [], []
#         global_description = self.query_data.get("global_description", "")
#         global_image_paths = self.query_data.get("global_images", [])
#
#         context_content, context_images = self._process_text_and_media(
#             global_description, global_image_paths
#         )
#         content.extend(context_content)
#         images.extend(context_images)
#
#         example_info = self.query_data["example_info"]
#         example_content, example_images = self.prepare_example_content(example_info)
#         content.extend(example_content)
#         images.extend(example_images)
#
#         return content, images
#
#     def prepare_example_content(self, example_info):
#         example_text = example_info["example_text"]
#         example_media_paths = example_info["image_paths"]
#         example_content, example_images = self._process_text_and_media(
#             example_text, example_media_paths, is_example=True
#         )
#
#         return example_content, example_images
#
#     def prepare_query_content(self, query_info):
#         query_text = query_info.get("query_text", "")
#         image_paths = query_info.get("image_paths", [])
#         query_content, query_images = self._process_text_and_media(
#             query_text, image_paths
#         )
#
#         return query_content, query_images
#
#     def query(self, task_name, query_data, position=0):
#         self.query_data = query_data
#
#         self._set_sampling_config(0)
#         context, context_image_list = self.prepare_context()
#         query_response = []
#
#         for query_idx, query_info in enumerate(
#             tqdm(
#                 query_data["queries"],
#                 desc=f"{task_name} - Processing queries",
#                 unit="query",
#                 position=position,
#             )
#         ):
#             exceed_image_quota = self._set_sampling_config(query_idx)
#
#             if not exceed_image_quota:
#                 images = []
#                 images.extend(context_image_list)
#
#                 query_content, query_images_list = self.prepare_query_content(
#                     query_info
#                 )
#
#                 if query_images_list:
#                     images.extend(query_images_list)
#
#                 image_tensors = process_images(
#                     images, self.image_processor, self.model.config
#                 )
#
#                 device = "cuda:0" if not hasattr(self, "device") else self.device
#                 image_tensors = [
#                     _image.to(dtype=torch.float16, device=device)
#                     for _image in image_tensors
#                 ]
#
#                 conv_template = "qwen_1_5"
#                 query_payload_list = context + query_content
#                 query_payload = "\n".join(query_payload_list)
#
#                 if self.print_response:
#                     print("Query information:", query_payload)
#                 conv = copy.deepcopy(conv_templates[conv_template])
#                 conv.append_message(conv.roles[0], query_payload)
#                 conv.append_message(conv.roles[1], None)
#                 prompt_question = conv.get_prompt()
#                 input_ids = tokenizer_image_token(
#                     prompt_question,
#                     self.tokenizer,
#                     IMAGE_TOKEN_INDEX,
#                     return_tensors="pt",
#                 ).unsqueeze(0)
#                 if hasattr(self, "device"):
#                     input_ids = input_ids.to(self.device)
#                 else:
#                     input_ids = input_ids.to("cuda:0")
#                 image_sizes = [image.size for image in images]
#
#                 max_new_tokens = self.config.get("session_len", 8192)
#
#                 import signal
#
#                 def handler(signum, frame):
#                     raise TimeoutError("Function call timed out")
#
#                 timeout = self.config.get("timeout", 60)
#                 signal.signal(signal.SIGALRM, handler)
#                 signal.alarm(timeout)
#
#                 try:
#                     cont = self.model.generate(
#                         input_ids,
#                         images=image_tensors,
#                         image_sizes=image_sizes,
#                         do_sample=False,
#                         temperature=0,
#                         max_new_tokens=max_new_tokens,
#                     )
#                     text_outputs = self.tokenizer.batch_decode(
#                         cont, skip_special_tokens=True
#                     )
#                     text_outputs = text_outputs[0]
#                 except TimeoutError as e:
#                     text_outputs = str(e)
#                 finally:
#                     signal.alarm(0)
#             else:
#                 text_outputs = (
#                     "Exceed the specified max number of images, skip running the model."
#                 )
#
#             if self.print_response:
#                 print("Model response:", text_outputs)
#             query_response.append(
#                 {
#                     "response": text_outputs,
#                     "correct_answer": query_info.get("correct_answer", None)
#                 }
#             )
#         return query_response

class Phi3v(InternVL):
    """
        Reuse all query preparation functions from InternVL,
        only need to implement the prompt composition here
        But we don't call InternVL's init function, which avoids loading the model
    """

    def __init__(
            self,
            api_key=None,
            model=None,
            query_data=None,
            resize=True,
            max_side=1000,
            print_response=False,
            max_num_image=1,
            system_message=None,
            total_demo_video_frames=4,
            **kwargs,
    ):
        super(OpenAI, self).__init__(
            api_key,
            model,
            query_data,
            resize,
            max_side,
            print_response,
            max_num_image,
            system_message,
            total_demo_video_frames,
            **kwargs,
        )

        # load config from config.json
        self.config = self.load_config(
            pathlib.Path(__file__).resolve().parent / "config.json"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        )

        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True, num_crops=4
        )

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _preprocess_payload(self, payload):
        image_count = 0

        def replace_image_placeholder(_):
            nonlocal image_count
            image_count += 1
            return f"<|image_{image_count}|>"

        composed_prompt = re.sub(r'<image>', replace_image_placeholder, payload)
        messages = [{
            "role": "user",
            "content": composed_prompt
        }, ]
        return messages

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        self._set_sampling_config(0)
        context, context_image_list = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
                tqdm(
                    query_data["queries"],
                    desc=f"{task_name} - Processing queries",
                    unit="query",
                    position=position,
                )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)
            if not exceed_image_quota:
                images = []
                query_content, query_images = self.prepare_query_content(query_info)
                if context_image_list:
                    images.extend(context_image_list)
                if query_images:
                    images.extend(query_images)

                query_payload_list = context + query_content
                query_payload = "\n".join(query_payload_list)
                messages = self._preprocess_payload(query_payload)

                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # inputs = self.processor(prompt, images, return_tensors="pt").to(
                #     "cuda:0"
                # )
                if images and len(images) > 0:
                    # 有图片的情况
                    inputs = self.processor(
                        prompt,
                        images=images,
                        return_tensors="pt"
                    ).to("cuda:0")
                else:
                    # 没有图片的情况（images=None或[]）
                    # 不要再传空列表，直接传 None 或者压根不传这个参数
                    inputs = self.processor(
                        prompt,
                        images=None,  # 或者干脆去掉 images 这个关键字
                        return_tensors="pt"
                    ).to("cuda:0")
                max_new_tokens = self.config.get("session_len", 8192)
                generation_args = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "do_sample": False,
                }

                import signal
                def handler(signum, frame):
                    raise TimeoutError("Function call timed out")

                timeout = self.config.get("timeout", 60)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)

                try:
                    generate_ids = self.model.generate(
                        **inputs,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        **generation_args,
                    )

                    # remove input tokens
                    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                    response = self.processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                except TimeoutError as e:
                    response = str(e)
                finally:
                    signal.alarm(0)

            else:
                response = (
                    "Exceed the specified max number of images, skip running the model."
                )

            if self.print_response:
                print(f"Model response:\n {response}")

            query_response.append(
                {
                    "response": response,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response

class Pixtral(Claude):

    def __init__(
        self,
        api_key=None,
        model=None,
        query_data=None,
        resize=True,
        max_side=1000,
        print_response=False,
        max_num_image=1,
        system_message: str | None = None,
        total_demo_video_frames=4,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            query_data=query_data,
            resize=resize,
            max_side=max_side,
            print_response=print_response,
            max_num_image=max_num_image,
            system_message=system_message,
            total_demo_video_frames=total_demo_video_frames,
            **kwargs,
        )
        self.kwargs = kwargs
        self.llm = LLM(**self.make_model_kwargs())
        self.ATTEMPT_LIMIT = 5

    def make_model_kwargs(self):
        """Set the LLM loading configuration."""
        kwargs = dict(
            model=self.model,
            max_num_seqs=5,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.max_num_image},
            tensor_parallel_size=self.kwargs.get("ngpus", 1),
            gpu_memory_utilization=self.kwargs.get("gpu_utils", 0.9),
            tokenizer_mode="mistral",
            max_num_batched_tokens=self.max_num_image * MAX_TOKENS_PER_IMAGE,
            max_model_len=4096,
        )

        num_gpus = torch.cuda.device_count()
        if num_gpus:
            kwargs["tensor_parallel_size"] = num_gpus

        return kwargs

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    def _prepare_data_with_lower_rate(self, query_info, query_idx, rate):
        _save = (
            self.max_num_image,
            self.total_demo_video_frames,
            self.max_demo_num_image,
        )
        self.max_num_image = math.floor(self.max_num_image / rate)
        self.total_demo_video_frames = math.floor(self.total_demo_video_frames / rate)
        self.max_demo_num_image = math.floor(self.max_demo_num_image / rate)
        self._set_sampling_config(query_idx)
        content = self.prepare_context()
        query_content = self.prepare_query_content(query_info)
        # recover the origial values
        self.max_num_image, self.total_demo_video_frames, self.max_demo_num_image = (
            _save
        )
        return content, query_content

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        self._set_sampling_config(0)

        context = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
            tqdm(
                query_data["queries"],
                desc=f"{task_name} - Processing queries",
                unit="query",
                position=position,
            )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)

            if not exceed_image_quota:
                query_content = self.prepare_query_content(query_info)
                text_content = None
                n_attempt = 0
                context_ = context  # avoid changing the context for later queries
                while text_content is None:
                    messages = self.prepare_system_message()
                    try:
                        messages.append(
                            {"role": "user", "content": context_ + query_content}
                        )
                        response = self.llm.chat(
                            messages, sampling_params=SAMPLING_PARAMS
                        )[0]
                        text_content = response.outputs[0].text
                        metrics = response.metrics
                        is_finished = response.finished
                    except ValueError as e:
                        logging.warning(
                            f"Error: {e}. Retry {n_attempt}/{self.ATTEMPT_LIMIT}..."
                        )
                        n_attempt += 1
                        context_, query_content = self._prepare_data_with_lower_rate(
                            query_info, query_idx, 1 + n_attempt * 0.2
                        )
                        if n_attempt > self.ATTEMPT_LIMIT:
                            text_content = str(e)
                            metrics = ""
                            is_finished = False

            else:
                text_content = (
                    "Exceed the specified max number of images, skip running the model."
                )
                metrics = ""
                is_finished = False

            if self.print_response:
                logging.info(
                    f"Model response: {text_content}\n Metrics: {metrics}\n is_finished: {is_finished}"
                )
            query_response.append(
                {
                    "response": text_content,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response



class Qwen2VL(OpenAI):
    def __init__(
        self,
        api_key=None,
        model=None,
        query_data=None,
        resize=True,
        max_side=1000,
        print_response=False,
        max_num_image=None,
        system_message=None,
        total_demo_video_frames=4,
        tp=None,
        **kwargs,
    ):
        super().__init__(
            api_key,
            model,
            query_data,
            resize,
            max_side,
            print_response,
            max_num_image,
            system_message,
            total_demo_video_frames,
            **kwargs,
        )
        # load config from config.json
        self.config = self.load_config(
            pathlib.Path(__file__).resolve().parent / "config.json"
        )

        self.processor = AutoProcessor.from_pretrained(model)

        self.llm = LLM(
            model=model,
            max_num_seqs=5,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max_num_image},
            tensor_parallel_size=kwargs.get("ngpus", 1),
            gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
        )

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_image_content(self, image_path):
        base64_image, mime_type = self.encode_image(image_path)
        # image format is modified
        return {
            "type": "image",
            "image": f"data:{mime_type};base64,{base64_image}",
        }

    def query(self, task_name, query_data, position=0):
        self.query_data = query_data
        self._set_sampling_config(0)
        context = self.prepare_context()

        query_response = []

        for query_idx, query_info in enumerate(
                tqdm(
                    query_data["queries"],
                    desc=f"{task_name} - Processing queries",
                    unit="query",
                    position=position,
                )
        ):
            exceed_image_quota = self._set_sampling_config(query_idx)

            query_content = self.prepare_query_content(query_info)

            # Check if the query_content length exceeds 32000 and truncate the excess part
            # if len(query_content) > 32000:
            #     logging.warning(f"Query content too long ({len(query_content)}). Truncating to 32000 characters.")
            #     query_content = query_content[len(query_content)-32000:]

            # print(query_content)

            if not exceed_image_quota:
                query_payload = [{"role": "user", "content": context + query_content}]
                text = self.processor.apply_chat_template(
                    query_payload, tokenize=False, add_generation_prompt=True
                )
                # print(text)
                image_inputs, _ = process_vision_info(query_payload)
                max_new_tokens = self.config.get("session_len", 8192)

                sampling_params = SamplingParams(
                    temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=None
                )
                try:
                    outputs = self.llm.generate(
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": image_inputs} if image_inputs is not None else None
                        },
                        sampling_params=sampling_params,
                    )
                    for o in outputs:
                        generated_text = o.outputs[0].text
                except:
                    generated_text = (
                        "Exceed the specified max number of tokens, skip running the model."
                    )
                    # print(generated_text)
            else:
                generated_text = (
                    "Exceed the specified max number of images, skip running the model."
                )

            if self.print_response:
                logging.info(f"Response: {generated_text}")

            query_response.append(
                {
                    "response": generated_text,
                    "correct_answer": query_info.get("correct_answer", None)
                }
            )

        return query_response
