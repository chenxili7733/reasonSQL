"""Batched model generation backends used by evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    batch_size: int = 8
    trust_remote_code: bool = True


def _render_chat_prompts(tokenizer: Any, messages: list[list[dict[str, str]]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]


def generate_vllm(
    model_path: str,
    messages: list[list[dict[str, str]]],
    config: GenerationConfig,
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vLLM evaluation requires the `eval` optional dependencies") from exc
    llm = LLM(
        model=model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=config.trust_remote_code,
        dtype="bfloat16",
    )
    prompts = _render_chat_prompts(llm.get_tokenizer(), messages)
    sampling = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    return [output.outputs[0].text for output in outputs]


def generate_transformers(
    model_path: str,
    messages: list[list[dict[str, str]]],
    config: GenerationConfig,
) -> list[str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Transformers evaluation requires the `eval` optional dependencies") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    responses: list[str] = []
    for start in range(0, len(messages), config.batch_size):
        batch = messages[start : start + config.batch_size]
        prompts = _render_chat_prompts(tokenizer, batch)
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_model_len - config.max_new_tokens,
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=max(config.temperature, 1e-6) if config.temperature > 0 else None,
                top_p=config.top_p if config.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[:, encoded["input_ids"].shape[1] :]
        responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return responses


def generate(
    model_path: str,
    messages: list[list[dict[str, str]]],
    config: GenerationConfig,
    backend: str = "vllm",
) -> list[str]:
    if backend == "vllm":
        return generate_vllm(model_path, messages, config)
    if backend == "transformers":
        return generate_transformers(model_path, messages, config)
    raise ValueError(f"unsupported generation backend: {backend}")

