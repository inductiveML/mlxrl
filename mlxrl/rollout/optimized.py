"""Optimized rollout variants with strict equivalence to the naive path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlxrl.rollout.naive import (
    Completion,
    SamplingConfig,
    decode_completion,
    encode_chat_prompt,
    eos_token_ids_from_tokenizer,
)


@dataclass(frozen=True)
class PrefixCache:
    """A materialized prompt KV cache plus first-step prompt logprobs."""

    cache: list[Any]
    first_logprobs: mx.array


def _clone_cache_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.array(value)
    return value


def clone_prompt_cache(prompt_cache: Sequence[Any]) -> list[Any]:
    """Clone a prompt cache so each group completion can mutate independently."""

    clones: list[Any] = []
    for cache in prompt_cache:
        state = tree_map(_clone_cache_value, cache.state)
        clones.append(type(cache).from_state(state, cache.meta_state))
    return clones


def prefill_prompt_once(model: nn.Module, prompt_tokens: Sequence[int]) -> PrefixCache:
    """Run the prompt forward pass once and materialize reusable KV state."""

    prompt_cache = make_prompt_cache(model)
    prompt_array = mx.array(list(prompt_tokens), dtype=mx.int32)
    logits = model(prompt_array[None], cache=prompt_cache)[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(  # Prefix sync: materialize reusable KV state and first-token logprobs.
        [cache.state for cache in prompt_cache],
        logprobs,
    )
    return PrefixCache(cache=prompt_cache, first_logprobs=logprobs)


def generate_from_prefix_cache(
    model: nn.Module,
    tokenizer: Any,
    prefix: PrefixCache,
    config: SamplingConfig,
    eos_token_ids: frozenset[int],
) -> tuple[tuple[int, ...], str]:
    """Decode one completion from a cloned prefix cache."""

    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    sampler = make_sampler(
        temp=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        top_k=config.top_k,
    )
    cache = clone_prompt_cache(prefix.cache)
    completion: list[int] = []

    next_token = sampler(prefix.first_logprobs)
    mx.eval(next_token)  # Rollout sync: Python needs the sampled token to append/EOS-check.
    token_id = int(next_token.item())
    completion.append(token_id)
    if token_id in eos_token_ids:
        return tuple(completion), decode_completion(tokenizer, completion)

    current = next_token
    for _ in range(1, config.max_tokens):
        logits = model(current[None], cache=cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        next_token = sampler(logprobs)
        mx.eval(next_token)  # Rollout sync: Python needs the sampled token to append/EOS-check.
        token_id = int(next_token.item())
        completion.append(token_id)
        if token_id in eos_token_ids:
            break
        current = next_token

    return tuple(completion), decode_completion(tokenizer, completion)


def generate_prefix_cached_group_rollouts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    group_size: int,
    config: SamplingConfig,
    seed: int | None = None,
    use_chat_template: bool = True,
) -> tuple[Completion, ...]:
    """Generate G completions per prompt after a single prompt prefill."""

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
        prefix = prefill_prompt_once(model, prompt_tokens)
        for group_index in range(group_size):
            completion_tokens, text = generate_from_prefix_cache(
                model=model,
                tokenizer=tokenizer,
                prefix=prefix,
                config=config,
                eos_token_ids=eos_ids,
            )
            completions.append(
                Completion(
                    prompt_index=prompt_index,
                    group_index=group_index,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    text=text,
                )
            )
    return tuple(completions)
