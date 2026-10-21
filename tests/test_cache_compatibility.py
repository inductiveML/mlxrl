from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ChunkedKVCache, KVCache

from mlxrl.rollout.optimized import (
    PrefixCache,
    _current_random_state,
    _set_random_state,
    fixed_decode_cache_from_prefix,
    fixed_decode_cache_from_prefixes,
)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("cache_type", [KVCache, ChunkedKVCache])
def test_fixed_decode_cache_uses_valid_prefix_not_allocated_capacity(cache_type) -> None:
    cache = cache_type() if cache_type is KVCache else cache_type(64)
    keys = mx.arange(48, dtype=mx.float32).reshape(1, 2, 3, 8)
    values = keys + 100
    cache.update_and_fetch(keys, values)

    (fixed,) = fixed_decode_cache_from_prefix([cache], max_tokens=4, batch_size=2)

    assert fixed.keys.shape == (2, 2, 7, 8)
    assert fixed.offset.tolist() == [3, 3]
    assert mx.array_equal(fixed.keys[:, :, :3, :], mx.broadcast_to(keys, (2, 2, 3, 8)))
    assert mx.array_equal(fixed.values[:, :, :3, :], mx.broadcast_to(values, (2, 2, 3, 8)))
    fixed.update_and_fetch(mx.ones((2, 2, 1, 8)), mx.zeros((2, 2, 1, 8)))
    assert fixed.offset.tolist() == [4, 4]
    assert cache.offset == 3
    assert cache.keys is not None and cache.values is not None
    assert mx.array_equal(cache.keys[:, :, :3, :], keys)
    assert mx.array_equal(cache.values[:, :, :3, :], values)


def test_batched_fixed_cache_preserves_different_prefix_lengths_and_values() -> None:
    prefixes = []
    for length in (2, 5):
        cache = KVCache()
        keys = mx.full((1, 2, length, 8), length, dtype=mx.float32)
        cache.update_and_fetch(keys, keys + 10)
        prefixes.append(PrefixCache(cache=[cache], first_logprobs=mx.zeros((1, 8))))

    (fixed,) = fixed_decode_cache_from_prefixes(prefixes, group_size=2, max_tokens=4)

    assert fixed.keys.shape == (4, 2, 9, 8)
    assert fixed.offset.tolist() == [2, 2, 5, 5]
    for start, length in ((0, 2), (2, 5)):
        assert mx.all(mx.equal(fixed.keys[start : start + 2, :, :length, :], length))
        assert mx.all(mx.equal(fixed.values[start : start + 2, :, :length, :], length + 10))
        assert mx.all(mx.equal(fixed.keys[start : start + 2, :, length:, :], 0))


def test_restoring_random_state_does_not_mutate_saved_snapshot() -> None:
    mx.random.seed(7)
    saved = _current_random_state()
    _set_random_state(mx.random.key(19))
    assert mx.array_equal(saved, mx.random.key(7))

    _set_random_state(saved)
    actual = mx.random.uniform(shape=(8,))
    mx.random.seed(7)
    expected = mx.random.uniform(shape=(8,))
    assert mx.array_equal(actual, expected)
