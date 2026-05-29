import os

os.environ.setdefault("VLLM_USE_V1", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from loguru import logger
from vllm import LLM, SamplingParams
from neurips25.tools import search_pubmed

# Base Llama-3.1-8B checkpoints (e.g. unsloth repack) ship without a chat template.
LLAMA31_INSTRUCT_TOKENIZER = "unsloth/Llama-3.1-8B-Instruct"


def _needs_llama31_instruct_tokenizer(model_name: str) -> bool:
    lower = model_name.lower()
    return "llama-3.1-8b" in lower and "instruct" not in lower

class BaseTextHFEval(BaseHuggingFaceEval):
    def __init__(self, tools=False, *args, **kwargs):
        """
        Initialize the BaseTextHFEval class with model name and system prompt.
        """
        super().__init__(*args, **kwargs)
        self.load_model()
        self.tools = []
        if tools:
            self.tools = [search_pubmed]

    def load_processor(self):
        pass

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token

    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        Text-only models ignore image file paths; inclusion is already in message content.
        """
        prompt = []
        for message in messages:
            prompt.append({"role": message["role"], "content": message["content"]})
        return prompt

    @logger.catch
    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        tools = self.tools if self.tools else None
        messages = self.convert_to_chat_format(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools,
        )
        inputs = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        return inputs

    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
            ).cpu()
        return (response, inputs["input_ids"].shape[1])

    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        (response, inputs) = response
        output_text = self.tokenizer.batch_decode(
            response[:,inputs:], skip_special_tokens=True
        )
        return output_text[0]

class BaseTextVLLMEval(BaseHuggingFaceEval):
    # Hancock text benchmarks: 8B models on a single A100 80GB.
    A100_8B_MAX_MODEL_LEN = 65536
    A100_8B_GPU_MEMORY_UTIL = 0.90
    A100_8B_MAX_OUTPUT_TOKENS = 512
    A100_8B_PROMPT_TOKEN_BUFFER = 1024

    def __init__(self, tools=False, use_all_but_last_device=False, score_choice_probs=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_all_but_last_device = use_all_but_last_device
        self.score_choice_probs = score_choice_probs
        self.load_model()
        self.tools = []
        if tools:
            self.tools = [search_pubmed]

    def load_processor(self):
        pass

    @staticmethod
    def _is_a100_8b_model(model_name: str) -> bool:
        """8B-class text models (Llama-3.1 / Meditron) on one A100 80GB."""
        name = model_name.lower()
        return any(tag in name for tag in ("8b", "7b", "meditron"))

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        to_quantize = [
            "Qwen/Qwen3-32B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen3-235B-A22B",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ]
        a100_8b = self._is_a100_8b_model(self.model_name)
        gpu_count = torch.cuda.device_count()
        llm_kwargs = dict(
            model=self.model_name,
            trust_remote_code=True,
            tokenizer_mode="mistral" if self.model_name.startswith("mistralai/Mistral-7B-Instruct") else "auto",
            config_format="mistral" if self.model_name.startswith("mistralai/Mistral-7B-Instruct") else "hf",
            load_format="mistral" if self.model_name.startswith("mistralai/") else "auto",
            max_num_seqs=1,
        )
        if a100_8b:
            # ~16 GB weights; 65k context KV cache ~8 GB — fits comfortably on A100 80GB.
            llm_kwargs.update(
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                enforce_eager=False,
                gpu_memory_utilization=self.A100_8B_GPU_MEMORY_UTIL,
                max_model_len=self.A100_8B_MAX_MODEL_LEN,
            )
            self._max_prompt_tokens = (
                self.A100_8B_MAX_MODEL_LEN
                - self.A100_8B_MAX_OUTPUT_TOKENS
                - self.A100_8B_PROMPT_TOKEN_BUFFER
            )
        else:
            llm_kwargs.update(
                dtype=torch.bfloat16 if self.model_name in to_quantize else "auto",
                enforce_eager=False,
                tensor_parallel_size=(
                    max(gpu_count - 1, 1) if self.use_all_but_last_device else gpu_count
                ),
                gpu_memory_utilization=0.95,
                max_model_len=32768,
            )
            self._max_prompt_tokens = 32768 - 1536 - 1024
        if self.model_name in to_quantize:
            llm_kwargs["quantization"] = "bitsandbytes"
        if self.model_name == "Qwen/Qwen3-235B-A22B":
            llm_kwargs["rope_scaling"] = {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
        if _needs_llama31_instruct_tokenizer(self.model_name):
            # Base weights + Instruct tokenizer/chat template (transformers >=4.44).
            llm_kwargs["tokenizer"] = LLAMA31_INSTRUCT_TOKENIZER
        self.model = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            max_tokens=self.A100_8B_MAX_OUTPUT_TOKENS if a100_8b else 1536,
            temperature=0.0,
        )

    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        Text-only models ignore image file paths; inclusion is already in message content.
        """
        prompt = []
        for message in messages:
            prompt.append({"role": message["role"], "content": message["content"]})
        return prompt

    def _count_chat_tokens(self, messages):
        tokenizer = self.model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    def _trim_messages(self, messages):
        """Drop oldest turns when Hancock case history exceeds the context window."""
        messages = self.convert_to_chat_format(messages)
        if not messages or self._count_chat_tokens(messages) <= self._max_prompt_tokens:
            return messages

        system = [messages[0]] if messages[0]["role"] == "system" else []
        rest = messages[1:] if system else list(messages)
        original_rest_len = len(rest)

        while len(rest) > 1 and self._count_chat_tokens(system + rest) > self._max_prompt_tokens:
            rest = rest[2:]

        if self._count_chat_tokens(system + rest) > self._max_prompt_tokens and rest:
            overflow = self._count_chat_tokens(system + rest) - self._max_prompt_tokens
            last = rest[-1]
            char_budget = max(len(last["content"]) - overflow * 4, 4000)
            rest[-1] = {
                "role": last["role"],
                "content": last["content"][-char_budget:],
            }

        if len(rest) < original_rest_len:
            logger.info(
                f"Trimmed conversation history ({original_rest_len} -> {len(rest)} turns, "
                f"~{self._count_chat_tokens(system + rest)} prompt tokens)"
            )
        return system + rest

    @logger.catch
    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        return self._trim_messages(messages)

    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        return self.model.chat(messages=inputs, sampling_params=self.sampling_params)

    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        return response[0].outputs[0].text

    def score_choice_probabilities(self, messages, choices):
        """
        Estimate per-choice probabilities via vLLM logprobs on the first generated token.
        """
        if not self.score_choice_probs:
            return None

        from neurips25.utils.confidence import normalize_choice_probs, logprob_to_prob

        scoring_messages = self._trim_messages(list(messages) + [{
            "role": "user",
            "content": (
                f"Based on all information above, reply with ONLY the single letter "
                f"of the best answer ({'/'.join(choices)}). No explanation."
            ),
        }])
        sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        try:
            outputs = self.model.chat(messages=scoring_messages, sampling_params=sampling)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during choice-probability scoring; skipping.")
            torch.cuda.empty_cache()
            return None
        logprobs_dict = outputs[0].outputs[0].logprobs
        if not logprobs_dict:
            return None

        first_token_logprobs = logprobs_dict[0]
        raw_probs = {}
        for choice in choices:
            best_logprob = None
            for token_id, logprob_entry in first_token_logprobs.items():
                token_text = logprob_entry.decoded_token.strip().upper()
                if token_text == choice or token_text.startswith(choice):
                    lp = logprob_entry.logprob
                    if best_logprob is None or lp > best_logprob:
                        best_logprob = lp
            if best_logprob is not None:
                raw_probs[choice] = logprob_to_prob(best_logprob)

        if not raw_probs:
            return None
        return normalize_choice_probs(raw_probs, choices)