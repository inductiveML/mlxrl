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


class FixedKVCache:
    """Fixed-capacity decode cache with explicit mutable array state."""

    def __init__(self, keys: mx.array, values: mx.array, offset: mx.array) -> None:
        self._state = [keys, values, offset]
        self.max_length = keys.shape[2]

    @property
    def keys(self) -> mx.array:
        return self._state[0]

    @property
    def values(self) -> mx.array:
        return self._state[1]

    @property
    def offset(self) -> mx.array:
        return self._state[2]

    @property
    def state(self) -> list[mx.array]:
        return self._state

    @state.setter
    def state(self, state: Sequence[mx.array]) -> None:
        self._state = list(state)
        self.max_length = self.keys.shape[2]

    @property
    def meta_state(self) -> tuple[str]:
        return (str(self.max_length),)

    @meta_state.setter
    def meta_state(self, meta_state: Sequence[str]) -> None:
        self.max_length = int(meta_state[0])

    @classmethod
    def from_state(
        cls,
        state: Sequence[mx.array],
        meta_state: Sequence[str],
    ) -> FixedKVCache:
        cache = cls(state[0], state[1], state[2])
        cache.meta_state = meta_state
        return cache

    def empty(self) -> bool:
        return False

    def size(self) -> int:
        return self.max_length

    def make_mask(
        self,
        n_tokens: int,
        return_array: bool = False,
        **_: Any,
    ) -> mx.array:
        del return_array
        positions = mx.arange(self.max_length)
        valid = positions[None, :] < (self.offset[:, None] + n_tokens)
        return valid[:, None, None, :]

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if keys.shape[2] != 1:
            raise ValueError("FixedKVCache only supports single-token decode updates.")

        positions = mx.arange(self.max_length)
        write = positions[None, None, :, None] == self.offset[:, None, None, None]
        self._state[0] = mx.where(write, keys, self.keys)
        self._state[1] = mx.where(write, values, self.values)
        self._state[2] = self.offset + keys.shape[2]
        return self.keys, self.values


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


def fixed_decode_cache_from_prefix(
    prompt_cache: Sequence[Any],
    max_tokens: int,
) -> list[FixedKVCache]:
    """Clone a prompt cache into fixed-size buffers for compiled decode."""

    fixed: list[FixedKVCache] = []
    for cache in prompt_cache:
        keys, values = cache.state
        prefix_length = keys.shape[2]
        max_length = prefix_length + max_tokens
        fixed_keys = mx.zeros(
            (keys.shape[0], keys.shape[1], max_length, keys.shape[3]),
            dtype=keys.dtype,
        )
        fixed_values = mx.zeros(
            (values.shape[0], values.shape[1], max_length, values.shape[3]),
            dtype=values.dtype,
        )
        fixed_keys[:, :, :prefix_length, :] = keys
        fixed_values[:, :, :prefix_length, :] = values
        fixed.append(
            FixedKVCache(
                keys=fixed_keys,
                values=fixed_values,
                offset=mx.array([prefix_length], dtype=mx.int32),
            )
        )
    mx.eval(  # Compile sync: materialize fixed KV buffers before state capture.
        [cache.state for cache in fixed],
    )
    return fixed


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
    compile_decode_step: bool = False,
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
    cache: list[Any]
    if compile_decode_step:
        cache = fixed_decode_cache_from_prefix(prefix.cache, config.max_tokens)
        decode_logprobs = _compiled_decode_logprobs(model, cache)
    else:
        cache = clone_prompt_cache(prefix.cache)
        decode_logprobs = _decode_logprobs(model, cache)
    completion: list[int] = []

    next_token = sampler(prefix.first_logprobs)
    mx.eval(next_token)  # Rollout sync: Python needs the sampled token to append/EOS-check.
    token_id = int(next_token.item())
    completion.append(token_id)
    if token_id in eos_token_ids:
        return tuple(completion), decode_completion(tokenizer, completion)

    current = next_token
    for _ in range(1, config.max_tokens):
        logprobs = decode_logprobs(current)
        next_token = sampler(logprobs)
        mx.eval(next_token)  # Rollout sync: Python needs the sampled token to append/EOS-check.
        token_id = int(next_token.item())
        completion.append(token_id)
        if token_id in eos_token_ids:
            break
        current = next_token

    return tuple(completion), decode_completion(tokenizer, completion)


def _decode_logprobs(model: nn.Module, cache: list[Any]) -> Any:
    def decode(token: mx.array) -> mx.array:
        logits = model(token[None], cache=cache)[:, -1, :]
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    return decode


def _compiled_decode_logprobs(model: nn.Module, cache: list[Any]) -> Any:
    cache_state = [layer_cache.state for layer_cache in cache]

    def decode(token: mx.array) -> mx.array:
        logits = model(token[None], cache=cache)[:, -1, :]
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    return mx.compile(decode, inputs=[model.state, cache_state], outputs=cache_state)


def generate_prefix_cached_group_rollouts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    group_size: int,
    config: SamplingConfig,
    seed: int | None = None,
    use_chat_template: bool = True,
    compile_decode_step: bool = False,
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
                compile_decode_step=compile_decode_step,
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
