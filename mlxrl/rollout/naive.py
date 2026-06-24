"""Naive one-sequence-at-a-time group rollouts using MLX-LM cache and sampling.

Attribution: uses MLX-LM cache construction and sampler APIs from
`mlx_lm.models.cache` and `mlx_lm.sample_utils` (MIT, Copyright Apple Inc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling knobs for the naive Phase 1 rollout path."""

    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 0


@dataclass(frozen=True)
class Completion:
    """One generated completion with token ids and decoded text."""

    prompt_index: int
    group_index: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    old_policy_logprobs: tuple[float, ...]
    text: str


def eos_token_ids_from_tokenizer(tokenizer: Any) -> frozenset[int]:
    """Collect EOS token ids exposed by common tokenizer wrappers."""

    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.add(int(eos))
    for attr in ("eos_token_ids", "stop_tokens", "stop_token_ids"):
        values = getattr(tokenizer, attr, None)
        if values is not None:
            ids.update(int(value) for value in values)
    return frozenset(ids)


def encode_chat_prompt(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool = True,
) -> tuple[int, ...]:
    """Apply a tokenizer chat template when available, then encode to ids."""

    text = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("Tokenizer produced no prompt tokens.")
    return tuple(int(token_id) for token_id in token_ids)


def decode_completion(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode completion tokens while tolerating tokenizer wrapper differences."""

    if not token_ids:
        return ""
    return str(tokenizer.decode(list(token_ids)))


def generate_completion(
    model: nn.Module,
    tokenizer: Any,
    prompt_tokens: Sequence[int],
    config: SamplingConfig,
    eos_token_ids: frozenset[int] | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    """Generate one completion with a fresh KV cache and no group sharing."""

    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    eos_ids = (
        eos_token_ids
        if eos_token_ids is not None
        else eos_token_ids_from_tokenizer(tokenizer)
    )
    sampler = make_sampler(
        temp=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        top_k=config.top_k,
    )
    prompt_cache = make_prompt_cache(model)
    current = mx.array(list(prompt_tokens), dtype=mx.int32)
    completion: list[int] = []
    old_policy_logprobs: list[float] = []

    for _ in range(config.max_tokens):
        logits = model(current[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        next_token = sampler(logprobs)
        sampled_logprob = sampled_token_logprobs(logprobs, next_token)
        mx.eval(  # Rollout sync: materialize sampled token/logprob for append/EOS-check.
            next_token,
            sampled_logprob,
        )
        token_id = int(next_token.item())
        completion.append(token_id)
        old_policy_logprobs.append(float(sampled_logprob.item()))
        if token_id in eos_ids:
            break
        current = next_token

    return (
        tuple(completion),
        tuple(old_policy_logprobs),
        decode_completion(tokenizer, completion),
    )


def sampled_token_logprobs(logprobs: mx.array, token_ids: mx.array) -> mx.array:
    """Gather raw model logprobs for sampled tokens."""

    return mx.squeeze(
        mx.take_along_axis(logprobs, token_ids[:, None], axis=-1),
        axis=-1,
    )


def generate_group_rollouts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    group_size: int,
    config: SamplingConfig,
    seed: int | None = None,
    use_chat_template: bool = True,
) -> tuple[Completion, ...]:
    """Generate G independent completions per prompt with fresh KV caches."""

    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if seed is not None:
        mx.random.seed(seed)

    eos_ids = eos_token_ids_from_tokenizer(tokenizer)
    completions: list[Completion] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_tokens = encode_chat_prompt(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
        )
        for group_index in range(group_size):
            completion_tokens, old_policy_logprobs, text = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                config=config,
                eos_token_ids=eos_ids,
            )
            completions.append(
                Completion(
                    prompt_index=prompt_index,
                    group_index=group_index,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    old_policy_logprobs=old_policy_logprobs,
                    text=text,
                )
            )
    return tuple(completions)
