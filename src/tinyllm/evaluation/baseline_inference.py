"""Deterministic Domain generation backend for the frozen M2.4c Baseline."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

from tinyllm.evaluation.baseline import (
    BaselineContractError,
    render_generation_prompt,
    score_domain_response,
)
from tinyllm.evaluation.baseline_runtime import (
    BaselinePreflightError,
    BaselineRuntime,
    BaselineRuntimeError,
)
from tinyllm.evaluation.baseline_schema import BaselineRunConfig, DomainItemResult
from tinyllm.evaluation.schema import EvaluationItem


@dataclass(frozen=True, slots=True)
class GeneratedResponse:
    """One decoded response and its exact Token accounting."""

    response: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: Literal["eos", "length"]


class DomainGenerationBackend(Protocol):
    """Minimal generation surface used by the contract-level runner and CPU tests."""

    def generate(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        max_sequence_length: int,
        max_new_tokens: int,
    ) -> tuple[GeneratedResponse, ...]:
        """Generate exactly one response per prompt in input order."""


def _normalize_eos_ids(eos_token_id: object) -> set[int]:
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, (list, tuple)) and all(
        isinstance(value, int) for value in eos_token_id
    ):
        return set(cast(Sequence[int], eos_token_id))
    raise BaselineRuntimeError("Tokenizer must define one or more integer EOS Token IDs")


class TransformersDomainBackend:
    """Lazy Transformers implementation loaded only after dependency and artifact checks."""

    def __init__(
        self,
        *,
        runtime: BaselineRuntime,
        config: BaselineRunConfig,
        model_path: Path,
        device: Literal["cpu", "cuda"],
    ) -> None:
        if not model_path.is_absolute() or not model_path.is_dir():
            raise BaselinePreflightError(
                "Baseline model path must be an existing absolute directory"
            )
        torch = runtime.torch
        if device == "cuda":
            if not bool(torch.cuda.is_available()):
                raise BaselinePreflightError("CUDA is unavailable for the Baseline")
            if not bool(torch.cuda.is_bf16_supported()):
                raise BaselinePreflightError("selected CUDA device does not support BF16")
        tokenizer = runtime.transformers.AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            raise BaselinePreflightError("Baseline Tokenizer does not define a padding Token")
        rendered_probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "TinyLLM template probe."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        expected_probe = (
            "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if rendered_probe != expected_probe:
            raise BaselinePreflightError("Baseline Tokenizer generation Template does not match")
        model = runtime.transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=config.model.attention_implementation,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self._device = device
        self._eos_ids = _normalize_eos_ids(tokenizer.eos_token_id)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        max_sequence_length: int,
        max_new_tokens: int,
    ) -> tuple[GeneratedResponse, ...]:
        """Generate greedy responses without silently truncating any frozen prompt."""

        torch = self._runtime.torch
        outputs: list[GeneratedResponse] = []
        for offset in range(0, len(prompts), batch_size):
            batch = list(prompts[offset : offset + batch_size])
            encoded = self._tokenizer(batch, padding=True, return_tensors="pt")
            prompt_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            if any(length > max_sequence_length for length in prompt_lengths):
                raise BaselineContractError("Domain prompt exceeds maximum sequence length")
            model_inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            input_width = int(model_inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = self._model.generate(
                    **model_inputs,
                    do_sample=False,
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=sorted(self._eos_ids),
                    return_dict_in_generate=True,
                )
            sequences = generated.sequences[:, input_width:].detach().cpu().tolist()
            for prompt_tokens, token_ids in zip(prompt_lengths, sequences, strict=True):
                effective = list(token_ids)
                finish_reason: Literal["eos", "length"] = "length"
                for index, token_id in enumerate(effective):
                    if token_id in self._eos_ids:
                        effective = effective[: index + 1]
                        finish_reason = "eos"
                        break
                response = str(
                    self._tokenizer.decode(
                        effective,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
                outputs.append(
                    GeneratedResponse(
                        response=response,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=len(effective),
                        finish_reason=finish_reason,
                    )
                )
        if len(outputs) != len(prompts):
            raise BaselineRuntimeError("generation backend returned the wrong response count")
        return tuple(outputs)


def seed_baseline(config: BaselineRunConfig, runtime: BaselineRuntime) -> None:
    """Seed Python, NumPy, PyTorch, and every visible CUDA generator."""

    random.seed(config.seeds.python)
    np.random.seed(config.seeds.numpy)
    runtime.torch.manual_seed(config.seeds.torch)
    if bool(runtime.torch.cuda.is_available()):
        runtime.torch.cuda.manual_seed_all(config.seeds.torch)


def run_domain_generation(
    config: BaselineRunConfig,
    items: Sequence[EvaluationItem],
    *,
    backend: DomainGenerationBackend,
) -> tuple[DomainItemResult, ...]:
    """Render, generate, and score the frozen Domain items without dropping raw outputs."""

    ordered_items = tuple(items)
    prompts = tuple(render_generation_prompt(item) for item in ordered_items)
    generated = backend.generate(
        prompts,
        batch_size=config.domain.batch_size,
        max_sequence_length=config.domain.max_sequence_length,
        max_new_tokens=config.domain.max_new_tokens,
    )
    if len(generated) != len(ordered_items):
        raise BaselineRuntimeError("generation backend returned the wrong response count")
    return tuple(
        score_domain_response(
            item,
            result.response,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            finish_reason=result.finish_reason,
        )
        for item, result in zip(ordered_items, generated, strict=True)
    )
