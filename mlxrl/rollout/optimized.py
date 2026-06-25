"""Optimized rollout variants with strict equivalence to the naive path.

Attribution: adapts MLX-LM KV-cache state conventions and sampling filters from
`mlx_lm.models.cache` and `mlx_lm.sample_utils` (MIT, Copyright Apple Inc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, make_sampler

from mlxrl.rollout.naive import (
    Completion,
    SamplingConfig,
    decode_completion,
    encode_chat_prompt,
    eos_token_ids_from_tokenizer,
    sampled_token_logprobs,
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
    batch_size: int = 1,
) -> list[FixedKVCache]:
    """Clone a prompt cache into fixed-size buffers for compiled decode."""

    fixed: list[FixedKVCache] = []
    for cache in prompt_cache:
        keys, values = cache.state
        prefix_length = keys.shape[2]
        max_length = prefix_length + max_tokens
        if keys.shape[0] != 1 and keys.shape[0] != batch_size:
            raise ValueError("Prefix cache batch size cannot be broadcast.")
        fixed_keys = mx.zeros(
            (batch_size, keys.shape[1], max_length, keys.shape[3]),
            dtype=keys.dtype,
        )
        fixed_values = mx.zeros(
            (batch_size, values.shape[1], max_length, values.shape[3]),
            dtype=values.dtype,
        )
        fixed_keys[:, :, :prefix_length, :] = mx.broadcast_to(
            keys,
            (batch_size, keys.shape[1], prefix_length, keys.shape[3]),
        )
        fixed_values[:, :, :prefix_length, :] = mx.broadcast_to(
            values,
            (batch_size, values.shape[1], prefix_length, values.shape[3]),
        )
        fixed.append(
            FixedKVCache(
                keys=fixed_keys,
                values=fixed_values,
                offset=mx.array([prefix_length] * batch_size, dtype=mx.int32),
            )
        )
    mx.eval(  # Compile sync: materialize fixed KV buffers before state capture.
        [cache.state for cache in fixed],
    )
    return fixed


def fixed_decode_cache_from_prefixes(
    prefixes: Sequence[PrefixCache],
    group_size: int,
    max_tokens: int,
) -> list[FixedKVCache]:
    """Build fixed decode buffers for all prompt groups in one batch."""

    if not prefixes:
        raise ValueError("At least one prefix cache is required.")
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")

    batch_size = len(prefixes) * group_size
    layer_count = len(prefixes[0].cache)
    fixed: list[FixedKVCache] = []
    for layer_index in range(layer_count):
        layer_states = [prefix.cache[layer_index].state for prefix in prefixes]
        first_keys, first_values = layer_states[0]
        max_prefix_length = max(keys.shape[2] for keys, _ in layer_states)
        max_length = max_prefix_length + max_tokens
        fixed_keys = mx.zeros(
            (batch_size, first_keys.shape[1], max_length, first_keys.shape[3]),
            dtype=first_keys.dtype,
        )
        fixed_values = mx.zeros(
            (
                batch_size,
                first_values.shape[1],
                max_length,
                first_values.shape[3],
            ),
            dtype=first_values.dtype,
        )
        offsets: list[int] = []
        for prompt_index, (keys, values) in enumerate(layer_states):
            prefix_length = keys.shape[2]
            start = prompt_index * group_size
            stop = start + group_size
            fixed_keys[start:stop, :, :prefix_length, :] = mx.broadcast_to(
                keys,
                (group_size, keys.shape[1], prefix_length, keys.shape[3]),
            )
            fixed_values[start:stop, :, :prefix_length, :] = mx.broadcast_to(
                values,
                (group_size, values.shape[1], prefix_length, values.shape[3]),
            )
            offsets.extend([prefix_length] * group_size)
        fixed.append(
            FixedKVCache(
                keys=fixed_keys,
                values=fixed_values,
                offset=mx.array(offsets, dtype=mx.int32),
            )
        )
    mx.eval(  # Compile sync: materialize prompt-set KV buffers before state capture.
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
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    """Decode one completion from a cloned prefix cache."""

    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    sampler = make_sampler(
        temp=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        top_k=config.top_k,
    )
    completion: list[int] = []
    old_policy_logprobs: list[float] = []

    next_token = sampler(prefix.first_logprobs)
    sampled_logprob = sampled_token_logprobs(prefix.first_logprobs, next_token)
    mx.eval(  # Rollout sync: materialize sampled token/logprob for append/EOS-check.
        next_token,
        sampled_logprob,
    )
    token_id = int(next_token.item())
    completion.append(token_id)
    old_policy_logprobs.append(float(sampled_logprob.item()))
    if token_id in eos_token_ids:
        return (
            tuple(completion),
            tuple(old_policy_logprobs),
            decode_completion(tokenizer, completion),
        )

    cache: list[Any]
    if compile_decode_step:
        cache = fixed_decode_cache_from_prefix(prefix.cache, config.max_tokens)
        decode_logprobs = _compiled_decode_logprobs(model, cache)
    else:
        cache = clone_prompt_cache(prefix.cache)
        decode_logprobs = _decode_logprobs(model, cache)

    current = next_token
    for _ in range(1, config.max_tokens):
        logprobs = decode_logprobs(current)
        next_token = sampler(logprobs)
        sampled_logprob = sampled_token_logprobs(logprobs, next_token)
        mx.eval(  # Rollout sync: materialize sampled token/logprob for append/EOS-check.
            next_token,
            sampled_logprob,
        )
        token_id = int(next_token.item())
        completion.append(token_id)
        old_policy_logprobs.append(float(sampled_logprob.item()))
        if token_id in eos_token_ids:
            break
        current = next_token

    return (
        tuple(completion),
        tuple(old_policy_logprobs),
        decode_completion(tokenizer, completion),
    )


def generate_group_from_prefix_cache(
    model: nn.Module,
    tokenizer: Any,
    prefix: PrefixCache,
    group_size: int,
    config: SamplingConfig,
    eos_token_ids: frozenset[int],
    compile_decode_step: bool = False,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...], str], ...]:
    """Decode a whole prompt group in one batched KV cache when safe."""

    return generate_prompt_set_from_prefix_caches(
        model=model,
        tokenizer=tokenizer,
        prefixes=[prefix],
        group_size=group_size,
        config=config,
        eos_token_ids=eos_token_ids,
        compile_decode_step=compile_decode_step,
    )


def generate_prompt_set_from_prefix_caches(
    model: nn.Module,
    tokenizer: Any,
    prefixes: Sequence[PrefixCache],
    group_size: int,
    config: SamplingConfig,
    eos_token_ids: frozenset[int],
    compile_decode_step: bool = False,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...], str], ...]:
    """Decode all prompt groups together with completion-major RNG keys."""

    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if not prefixes:
        raise ValueError("At least one prefix cache is required.")

    row_count = len(prefixes) * group_size
    start_random_state = _current_random_state()
    draw_keys: list[list[mx.array]] | None = None
    final_random_state: mx.array | None = None
    if _uses_random_sampling(config):
        draw_keys, final_random_state = _completion_major_draw_keys(
            start_random_state,
            group_size=row_count,
            max_tokens=config.max_tokens,
        )

    first_logprobs = mx.concatenate(
        [
            mx.broadcast_to(
                prefix.first_logprobs,
                (group_size, prefix.first_logprobs.shape[-1]),
            )
            for prefix in prefixes
        ],
        axis=0,
    )

    rows: list[list[int]] = [[] for _ in range(row_count)]
    old_policy_logprob_rows: list[list[float]] = [[] for _ in range(row_count)]
    active = [True] * row_count
    early_eos = False
    next_token = _sample_batched_logprobs(
        first_logprobs,
        config,
        None if draw_keys is None else [row[0] for row in draw_keys],
    )
    sampled_logprob = sampled_token_logprobs(first_logprobs, next_token)
    mx.eval(  # Rollout sync: materialize sampled tokens/logprobs for append/EOS-check.
        next_token,
        sampled_logprob,
    )
    current = next_token
    for row_index, (token_id, logprob) in enumerate(
        zip(_token_ids(next_token), _logprob_values(sampled_logprob), strict=True)
    ):
        rows[row_index].append(token_id)
        old_policy_logprob_rows[row_index].append(logprob)
        if token_id in eos_token_ids:
            active[row_index] = False
            early_eos = config.max_tokens > 1

    if early_eos and draw_keys is not None:
        _set_random_state(start_random_state)
        return _generate_prompt_set_sequentially(
            model=model,
            tokenizer=tokenizer,
            prefixes=prefixes,
            group_size=group_size,
            config=config,
            eos_token_ids=eos_token_ids,
            compile_decode_step=compile_decode_step,
        )
    if not any(active):
        return _rollout_rows_to_outputs(tokenizer, rows, old_policy_logprob_rows)

    cache: list[Any] = fixed_decode_cache_from_prefixes(
        prefixes,
        group_size=group_size,
        max_tokens=config.max_tokens,
    )
    decode_logprobs = (
        _compiled_decode_logprobs(model, cache)
        if compile_decode_step
        else _decode_logprobs(model, cache)
    )

    for step in range(1, config.max_tokens):
        logprobs = decode_logprobs(current)
        next_token = _sample_batched_logprobs(
            logprobs,
            config,
            None if draw_keys is None else [row[step] for row in draw_keys],
        )
        sampled_logprob = sampled_token_logprobs(logprobs, next_token)
        mx.eval(  # Rollout sync: materialize sampled tokens/logprobs for append/EOS-check.
            next_token,
            sampled_logprob,
        )
        for row_index, (token_id, logprob) in enumerate(
            zip(_token_ids(next_token), _logprob_values(sampled_logprob), strict=True)
        ):
            if not active[row_index]:
                continue
            rows[row_index].append(token_id)
            old_policy_logprob_rows[row_index].append(logprob)
            if token_id in eos_token_ids:
                active[row_index] = False
                early_eos = step < config.max_tokens - 1
        current = next_token
        if early_eos and draw_keys is not None:
            _set_random_state(start_random_state)
            return _generate_prompt_set_sequentially(
                model=model,
                tokenizer=tokenizer,
                prefixes=prefixes,
                group_size=group_size,
                config=config,
                eos_token_ids=eos_token_ids,
                compile_decode_step=compile_decode_step,
            )
        if not any(active):
            break

    if final_random_state is not None:
        _set_random_state(final_random_state)

    return _rollout_rows_to_outputs(tokenizer, rows, old_policy_logprob_rows)


def _generate_prompt_set_sequentially(
    *,
    model: nn.Module,
    tokenizer: Any,
    prefixes: Sequence[PrefixCache],
    group_size: int,
    config: SamplingConfig,
    eos_token_ids: frozenset[int],
    compile_decode_step: bool,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...], str], ...]:
    outputs: list[tuple[tuple[int, ...], tuple[float, ...], str]] = []
    for prefix in prefixes:
        outputs.extend(
            generate_from_prefix_cache(
                model=model,
                tokenizer=tokenizer,
                prefix=prefix,
                config=config,
                eos_token_ids=eos_token_ids,
                compile_decode_step=compile_decode_step,
            )
            for _ in range(group_size)
        )
    return tuple(outputs)


def _rollout_rows_to_outputs(
    tokenizer: Any,
    rows: Sequence[Sequence[int]],
    old_policy_logprob_rows: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[int, ...], tuple[float, ...], str], ...]:
    return tuple(
        (
            tuple(row),
            tuple(old_policy_logprobs),
            decode_completion(tokenizer, row),
        )
        for row, old_policy_logprobs in zip(
            rows,
            old_policy_logprob_rows,
            strict=True,
        )
    )


def _decode_logprobs(model: nn.Module, cache: list[Any]) -> Any:
    def decode(token: mx.array) -> mx.array:
        logits = model(token[:, None], cache=cache)[:, -1, :]
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    return decode


def _compiled_decode_logprobs(model: nn.Module, cache: list[Any]) -> Any:
    cache_state = [layer_cache.state for layer_cache in cache]

    def decode(token: mx.array) -> mx.array:
        logits = model(token[:, None], cache=cache)[:, -1, :]
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    return mx.compile(decode, inputs=[model.state, cache_state], outputs=cache_state)


def _uses_random_sampling(config: SamplingConfig) -> bool:
    return config.temperature != 0


def _current_random_state() -> mx.array:
    return mx.array(cast(list[mx.array], mx.random.state)[0])


def _set_random_state(state: mx.array) -> None:
    cast(list[mx.array], mx.random.state)[0] = state


def _token_ids(tokens: mx.array) -> list[int]:
    values = cast(Sequence[Any], tokens.tolist())
    return [int(token) for token in values]


def _logprob_values(logprobs: mx.array) -> list[float]:
    values = cast(Sequence[Any], logprobs.tolist())
    return [float(value) for value in values]


def _completion_major_draw_keys(
    start_state: mx.array,
    group_size: int,
    max_tokens: int,
) -> tuple[list[list[mx.array]], mx.array]:
    state = start_state
    keys: list[list[mx.array]] = []
    for _ in range(group_size):
        row: list[mx.array] = []
        for _ in range(max_tokens):
            split = mx.random.split(state, 2)
            state = split[0]
            row.append(split[1])
        keys.append(row)
    return keys, state


def _filtered_logprobs(logprobs: mx.array, config: SamplingConfig) -> mx.array:
    if 0 < config.top_p < 1.0:
        logprobs = apply_top_p(logprobs, config.top_p)
    if config.min_p != 0.0:
        logprobs = apply_min_p(logprobs, config.min_p, 1)
    if config.top_k > 0:
        logprobs = apply_top_k(logprobs, config.top_k)
    return logprobs


def _sample_batched_logprobs(
    logprobs: mx.array,
    config: SamplingConfig,
    keys: Sequence[mx.array] | None,
) -> mx.array:
    filtered = _filtered_logprobs(logprobs, config)
    if not _uses_random_sampling(config):
        return mx.argmax(filtered, axis=-1)
    if keys is None:
        return mx.random.categorical(filtered * (1 / config.temperature))
    samples = [
        mx.random.categorical(
            filtered[index : index + 1] * (1 / config.temperature),
            key=key,
        )
        for index, key in enumerate(keys)
    ]
    return mx.concatenate(samples, axis=0)


def generate_prefix_cached_group_rollouts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    group_size: int,
    config: SamplingConfig,
    seed: int | None = None,
    use_chat_template: bool = True,
    compile_decode_step: bool = False,
    batch_groups: bool = False,
) -> tuple[Completion, ...]:
    """Generate G completions per prompt after a single prompt prefill."""

    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if seed is not None:
        mx.random.seed(seed)

    eos_ids = eos_token_ids_from_tokenizer(tokenizer)
    completions: list[Completion] = []
    prompt_tokens_by_index: list[tuple[int, ...]] = []
    prefixes: list[PrefixCache] = []
    for prompt in prompts:
        prompt_tokens = encode_chat_prompt(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
        )
        prompt_tokens_by_index.append(prompt_tokens)
        prefixes.append(prefill_prompt_once(model, prompt_tokens))

    if batch_groups:
        batched_outputs = generate_prompt_set_from_prefix_caches(
            model=model,
            tokenizer=tokenizer,
            prefixes=prefixes,
            group_size=group_size,
            config=config,
            eos_token_ids=eos_ids,
            compile_decode_step=compile_decode_step,
        )
        for row_index, (completion_tokens, old_policy_logprobs, text) in enumerate(
            batched_outputs
        ):
            prompt_index = row_index // group_size
            group_index = row_index % group_size
            completions.append(
                Completion(
                    prompt_index=prompt_index,
                    group_index=group_index,
                    prompt_tokens=prompt_tokens_by_index[prompt_index],
                    completion_tokens=completion_tokens,
                    old_policy_logprobs=old_policy_logprobs,
                    text=text,
                )
            )
        return tuple(completions)

    for prompt_index, (prompt_tokens, prefix) in enumerate(
        zip(prompt_tokens_by_index, prefixes, strict=True)
    ):
        group_outputs = tuple(
            generate_from_prefix_cache(
                model=model,
                tokenizer=tokenizer,
                prefix=prefix,
                config=config,
                eos_token_ids=eos_ids,
                compile_decode_step=compile_decode_step,
            )
            for _ in range(group_size)
        )
        for group_index, (completion_tokens, old_policy_logprobs, text) in enumerate(
            group_outputs
        ):
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
