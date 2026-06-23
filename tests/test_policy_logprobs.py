from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlxrl.policy.logprobs import completion_logprobs, prefix_cached_completion_logprobs


class ToyCache:
    def __init__(self, sums: mx.array | None = None) -> None:
        self.sums = sums if sums is not None else mx.zeros((1,), dtype=mx.int32)

    @property
    def state(self) -> tuple[mx.array]:
        return (self.sums,)

    @state.setter
    def state(self, state: Sequence[mx.array]) -> None:
        self.sums = state[0]

    @property
    def meta_state(self) -> tuple[()]:
        return ()

    @meta_state.setter
    def meta_state(self, meta_state: Sequence[str]) -> None:
        del meta_state

    @classmethod
    def from_state(
        cls,
        state: Sequence[mx.array],
        meta_state: Sequence[str],
    ) -> ToyCache:
        del meta_state
        return cls(state[0])

    @classmethod
    def merge(cls, caches: Sequence[ToyCache]) -> ToyCache:
        return cls(mx.concatenate([cache.sums for cache in caches], axis=0))


class ToyCachedModel(nn.Module):
    vocab_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.cached_call_shapes: list[tuple[int, int]] = []

    def make_cache(self) -> list[ToyCache]:
        return [ToyCache()]

    def __call__(
        self,
        tokens: mx.array,
        cache: Sequence[ToyCache] | None = None,
    ) -> mx.array:
        batch_size, sequence_length = tokens.shape
        token_values = tokens.astype(mx.float32)
        base = (
            cache[0].sums.astype(mx.float32)[:, None]
            if cache is not None
            else mx.zeros((batch_size, 1), dtype=mx.float32)
        )
        cumulative = base + mx.cumsum(token_values, axis=1)
        if cache is not None:
            self.cached_call_shapes.append((batch_size, sequence_length))
            cache[0].sums = (base[:, 0] + mx.sum(tokens, axis=1)).astype(mx.int32)
        vocab = mx.arange(self.vocab_size, dtype=mx.float32)
        return cumulative[:, :, None] * (vocab[None, None, :] + 1.0) / 100.0


def test_prefix_cached_completion_logprobs_match_full_forward_on_toy_model() -> None:
    model = ToyCachedModel()
    prompts = ((2, 3, 4), (2, 3, 4))
    completions = ((5, 6), (7,))

    full = completion_logprobs(model, prompts, completions, pad_token_id=0)
    cached = prefix_cached_completion_logprobs(model, prompts, completions, pad_token_id=0)
    max_error = mx.max(mx.abs(full.logprobs - cached.logprobs))
    mask_error = mx.max(mx.abs(full.mask - cached.mask))
    mx.eval(  # Test sync: materialize logprob/mask comparisons for assertions.
        max_error,
        mask_error,
    )

    assert float(max_error.item()) < 1e-6
    assert float(mask_error.item()) == 0.0
    assert model.cached_call_shapes == [(1, 2), (2, 2)]
