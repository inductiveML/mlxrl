"""Focused tests for Phase 2 rollout helpers that do not download a model."""

from __future__ import annotations

import mlx.core as mx
import pytest

import mlxrl.rollout.optimized as optimized_module
from mlxrl.rollout.naive import SamplingConfig
from mlxrl.rollout.optimized import (
    FixedKVCache,
    PrefixCache,
    _completion_major_draw_keys,
    _set_random_state,
    generate_prompt_set_from_prefix_caches,
)

pytestmark = pytest.mark.metal


class ToyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def test_completion_major_draw_keys_match_mlx_global_stream() -> None:
    logits = mx.array([[0.1, 0.2, 1.7, -0.3]], dtype=mx.float32)

    mx.random.seed(7)
    global_draws: list[int] = []
    for _ in range(6):
        token = mx.random.categorical(logits)
        mx.eval(token)  # Test sync: materialize sampled token for Python assertion.
        global_draws.append(int(token.item()))

    keys, final_state = _completion_major_draw_keys(
        mx.random.key(7),
        group_size=2,
        max_tokens=3,
    )
    explicit_draws: list[int] = []
    for row in keys:
        for key in row:
            token = mx.random.categorical(logits, key=key)
            mx.eval(token)  # Test sync: materialize sampled token for Python assertion.
            explicit_draws.append(int(token.item()))

    assert explicit_draws == global_draws

    _set_random_state(final_state)
    explicit_next = mx.random.categorical(logits)
    mx.eval(explicit_next)  # Test sync: materialize next explicit-state draw.
    mx.random.seed(7)
    for _ in range(6):
        token = mx.random.categorical(logits)
        mx.eval(token)  # Test sync: advance and materialize consumed global draw.
    global_next = mx.random.categorical(logits)
    mx.eval(global_next)  # Test sync: materialize next global draw for assertion.
    assert int(explicit_next.item()) == int(global_next.item())


def test_fixed_kv_cache_updates_rows_at_dynamic_offsets() -> None:
    cache = FixedKVCache(
        keys=mx.zeros((2, 1, 4, 2), dtype=mx.float32),
        values=mx.zeros((2, 1, 4, 2), dtype=mx.float32),
        offset=mx.array([1, 2], dtype=mx.int32),
    )
    keys = mx.array([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=mx.float32)
    values = mx.array([[[[5.0, 6.0]]], [[[7.0, 8.0]]]], dtype=mx.float32)

    updated_keys, updated_values = cache.update_and_fetch(keys, values)
    mask = cache.make_mask(1)
    mx.eval(  # Test sync: materialize cache arrays and mask for Python assertions.
        updated_keys,
        updated_values,
        cache.offset,
        mask,
    )

    assert cache.offset.tolist() == [2, 3]
    assert updated_keys[0, 0, 1].tolist() == [1.0, 2.0]
    assert updated_keys[1, 0, 2].tolist() == [3.0, 4.0]
    assert updated_values[0, 0, 1].tolist() == [5.0, 6.0]
    assert updated_values[1, 0, 2].tolist() == [7.0, 8.0]
    assert mask.shape == (2, 1, 1, 4)


def test_greedy_batched_rollout_returns_first_token_eos_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = PrefixCache(
        cache=[],
        first_logprobs=mx.array([[0.0, -10.0]], dtype=mx.float32),
    )

    def fail_fallback(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("greedy early-EOS rollout should not fallback")

    monkeypatch.setattr(optimized_module, "generate_from_prefix_cache", fail_fallback)

    outputs = generate_prompt_set_from_prefix_caches(
        model=object(),  # type: ignore[arg-type]
        tokenizer=ToyTokenizer(),
        prefixes=[prefix],
        group_size=2,
        config=SamplingConfig(max_tokens=4, temperature=0.0),
        eos_token_ids=frozenset({0}),
    )

    assert outputs == (
        ((0,), (0.0,), "0"),
        ((0,), (0.0,), "0"),
    )


def test_random_batched_rollout_keeps_fallback_on_first_token_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = PrefixCache(
        cache=[],
        first_logprobs=mx.array([[0.0]], dtype=mx.float32),
    )
    calls = 0

    def fake_fallback(*args: object, **kwargs: object):
        nonlocal calls
        del args, kwargs
        calls += 1
        return (99,), (-0.5,), "fallback"

    monkeypatch.setattr(optimized_module, "generate_from_prefix_cache", fake_fallback)

    outputs = generate_prompt_set_from_prefix_caches(
        model=object(),  # type: ignore[arg-type]
        tokenizer=ToyTokenizer(),
        prefixes=[prefix],
        group_size=2,
        config=SamplingConfig(max_tokens=4, temperature=0.7),
        eos_token_ids=frozenset({0}),
    )

    assert calls == 2
    assert outputs == (
        ((99,), (-0.5,), "fallback"),
        ((99,), (-0.5,), "fallback"),
    )
