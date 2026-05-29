import os

os.environ.setdefault("VLLM_USE_V1", "0")

import torch
import base64
from typing import NamedTuple, Optional
from dataclasses import asdict
import PIL
from transformers import AutoProcessor
from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from loguru import logger
from vllm import LLM, SamplingParams, EngineArgs
from vllm.lora.request import LoRARequest
from neurips25.tools import search_pubmed

if os.environ.get("VLLM_USE_V1", "1") == "0":
    print("WARNING: Using VLLM V0 engine.")

def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"

def _convert_to_chat_format(messages):
    """
    Convert the input messages to the chat format required by the model.
    """
    prompt = []
    all_image_urls = []
    for message in messages:
        role = message["role"]
        content_list = []
        if "files" in message:
            for file in message["files"]:
                data_url = file_to_data_url(file)
                content_list.append({"type": "image_url", "image_url": {"url": data_url}})
                all_image_urls.append(data_url)
        content_list.append({"type": "text", "text": message["content"]})
        prompt.append({"role": role, "content": content_list})
    return prompt, all_image_urls

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[PIL.Image.Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None

class MistralVLLMEval(BaseHuggingFaceEval):
    def __init__(self, tools=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_processor()
        self.load_model()

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        self.engine_args = EngineArgs(
            model=self.model_name,
            max_num_seqs=1,
            trust_remote_code=True,
            # tokenizer_mode="mistral" if self.model_name.startswith("mistralai/") else "auto",
            # config_format="mistral" if self.model_name.startswith("mistralai/") else "hf",
            # load_format="mistral" if self.model_name.startswith("mistralai/") else "auto",
            limit_mm_per_prompt={"image": 999},
            # mm_processor_kwargs={"max_dynamic_patch": 4},
            tensor_parallel_size=torch.cuda.device_count(),

        )
        self.engine_args = asdict(self.engine_args)
        self.model = LLM(**self.engine_args)
    
    def convert_to_chat_format(self, messages):
        return _convert_to_chat_format(messages)

    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        messages, image_urls = self.convert_to_chat_format(messages)
        req_data = ModelRequestData(
            engine_args=self.engine_args,
            prompt=None,
            image_data=[url for url in image_urls],
        )
        return req_data, messages

    def generate_response(self, pair):
        """
        Generate a response from the model based on the inputs.
        """
        req_data, messages = pair
        sampling_params = SamplingParams(
            max_tokens=1536,
            temperature=0.0,
            stop_token_ids=req_data.stop_token_ids,
        )
        return self.model.chat(
            messages=messages, 
            sampling_params=sampling_params,
            chat_template=req_data.chat_template,
            lora_request=req_data.lora_requests,
        )

    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        return response[0].outputs[0].text

class LlamaVLLMEval(BaseHuggingFaceEval):
    def __init__(self, tools=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_processor()
        self.load_model()

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        self.engine_args = EngineArgs(
            model=self.model_name,
            max_num_seqs=1,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 20} if self.model_name.endswith("11B-Vision-Instruct") else {"image": 12},
            max_model_len=70000 if self.model_name.endswith("11B-Vision-Instruct") else 100000,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            enforce_eager=False,
        )
        self.engine_args = asdict(self.engine_args)
        self.model = LLM(**self.engine_args)

    def convert_to_chat_format(self, messages):
        return _convert_to_chat_format(messages)

    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        messages, image_urls = self.convert_to_chat_format(messages)
        req_data = ModelRequestData(
            engine_args=self.engine_args,
            prompt=None,
            image_data=[url for url in image_urls],
        )
        return req_data, messages

    def generate_response(self, pair):
        """
        Generate a response from the model based on the inputs.
        """
        req_data, messages = pair
        sampling_params = SamplingParams(
            max_tokens=1536,
            temperature=0.0,
            stop_token_ids=req_data.stop_token_ids,
        )
        return self.model.chat(
            messages=messages, 
            sampling_params=sampling_params,
            chat_template=req_data.chat_template,
            lora_request=req_data.lora_requests,
        )

    def decode_response(self, response):
        return response[0].outputs[0].text

