import os

# Must be set before vLLM is imported (numba/numpy compatibility on v1 engine)
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from neurips25.eval import *
from omegaconf import OmegaConf

# Unsloth BF16 repack; use Instruct weights for agent benchmarks (base model echoes prompts).
LLAMA31_8B_UNSLOTH = "unsloth/Llama-3.1-8B-Instruct"
LLAMA31_8B_UNSLOTH_BASE = "unsloth/Meta-Llama-3.1-8B"


def _resolve_llama31_model(model_name: str) -> str:
    """Base Llama-3.1-8B cannot follow [ANSWER]/[REQUEST] tags; route to Instruct."""
    if model_name in (LLAMA31_8B_UNSLOTH_BASE, "meta-llama/Llama-3.1-8B"):
        print(
            f"Note: {model_name} is a base (non-instruct) checkpoint; "
            f"using {LLAMA31_8B_UNSLOTH} for agent evaluation."
        )
        return LLAMA31_8B_UNSLOTH
    return model_name


def get_model(model_name, score_choice_probs=True):
    model_name = _resolve_llama31_model(model_name)
    if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        return MistralVLLMEval(model_name=model_name), "mistralsmall"
    if model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
        return LlamaVLLMEval(model_name=model_name), "llama90b"
    elif "gpt-4o" in model_name.lower():
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        openai_token = conf.openai_token
        return GPT4oEval(model_name=model_name, openai_token=openai_token), "gpt-4o"
    elif "Qwen2.5-VL" in model_name:
        return Qwen25VLEval(model_name=model_name), "qwen2.5-vl"
    elif "Qwen3" in model_name:
        return BaseTextVLLMEval(model_name=model_name, score_choice_probs=score_choice_probs), "qwen3"
    elif "Meditron" in model_name or "meditron" in model_name.lower():
        return BaseTextVLLMEval(model_name=model_name, score_choice_probs=score_choice_probs), "meditron3-8b"
    elif model_name == LLAMA31_8B_UNSLOTH or (
        "Llama-3" in model_name and ("8B" in model_name or "8b" in model_name)
    ):
        return BaseTextVLLMEval(model_name=model_name, score_choice_probs=score_choice_probs), "llama31-8b"
    elif "Llama-3" in model_name:
        return BaseTextVLLMEval(model_name=model_name, score_choice_probs=score_choice_probs), "llama-3"
    elif "gemma-3" in model_name.lower():
        return Gemma3Eval(model_name=model_name), "gemma-3"
    elif "internvl3" in model_name.lower():
        return InternVLEval(model_name=model_name), "internvl3"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    