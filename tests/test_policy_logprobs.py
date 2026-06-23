from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlxrl.policy.logprobs import (
    completion_logprobs,
    dual_logprobs,
    prefix_cached_completion_logprobs,
)
from mlxrl.rollout.naive import Completion, SamplingConfig
from mlxrl.rollout.optimized import generate_from_prefix_cache, prefill_prompt_once
from mlxrl.train.grpo import batch_from_rollouts, old_policy_logprobs_from_rollouts


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


class ToyTokenizer:
    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


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


def test_rollout_captured_logprobs_match_completion_forward_on_toy_model() -> None:
    model = ToyCachedModel()
    tokenizer = ToyTokenizer()
    prompt = (2, 3, 4)
    prefix = prefill_prompt_once(model, prompt)

    completion_tokens, captured_logprobs, _ = generate_from_prefix_cache(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        config=SamplingConfig(max_tokens=2, temperature=0.0),
        eos_token_ids=frozenset(),
        compile_decode_step=False,
    )
    full = completion_logprobs(model, (prompt,), (completion_tokens,), pad_token_id=0)
    captured = mx.array([captured_logprobs], dtype=full.logprobs.dtype)
    max_error = mx.max(mx.abs(full.logprobs - captured))
    mx.eval(  # Test sync: materialize captured/full logprob comparison.
        max_error,
    )

    assert float(max_error.item()) < 1e-6


def test_dual_logprobs_can_skip_reference_forward_for_zero_beta() -> None:
    model = ToyCachedModel()
    prompts = ((2, 3, 4), (2, 3, 4))
    completions = ((5, 6), (7,))

    dual = dual_logprobs(
        model,
        prompts,
        completions,
        pad_token_id=0,
        compute_reference=False,
    )
    max_error = mx.max(mx.abs(dual.policy - dual.reference))
    mx.eval(  # Test sync: materialize policy/reference comparison.
        max_error,
        dual.mask,
    )

    assert float(max_error.item()) == 0.0
    assert dual.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert model.cached_call_shapes == []


def test_old_policy_logprobs_from_rollouts_pad_to_completion_mask() -> None:
    completions = (
        Completion(
            prompt_index=0,
            group_index=0,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(5, 6),
            old_policy_logprobs=(-1.0, -2.0),
            text="5 6",
        ),
        Completion(
            prompt_index=0,
            group_index=1,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(7,),
            old_policy_logprobs=(-3.0,),
            text="7",
        ),
    )

    old_policy = old_policy_logprobs_from_rollouts(completions)
    mx.eval(  # Test sync: materialize padded rollout logprobs and mask.
        old_policy.logprobs,
        old_policy.mask,
    )

    assert old_policy.logprobs.tolist() == [[-1.0, -2.0], [-3.0, 0.0]]
    assert old_policy.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_batch_from_rollouts_recomputes_old_policy_logprobs() -> None:
    model = ToyCachedModel()
    completions = (
        Completion(
            prompt_index=0,
            group_index=0,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(5, 6),
            old_policy_logprobs=(-999.0, -999.0),
            text="5 6",
        ),
        Completion(
            prompt_index=0,
            group_index=1,
            prompt_tokens=(2, 3, 4),
            completion_tokens=(7,),
            old_policy_logprobs=(-999.0,),
            text="7",
        ),
    )
    expected = completion_logprobs(
        model,
        tuple(completion.prompt_tokens for completion in completions),
        tuple(completion.completion_tokens for completion in completions),
        pad_token_id=0,
    )

    batch = batch_from_rollouts(
        model=model,
        completions=completions,
        rewards=(1.0, 0.0),
        group_size=2,
        pad_token_id=0,
    )
    mx.eval(  # Test sync: materialize batch logprobs/mask for assertions.
        batch.old_policy_logprobs,
        batch.mask,
        expected.logprobs,
    )

    max_error = mx.max(mx.abs(batch.old_policy_logprobs - expected.logprobs))
    mx.eval(max_error)  # Test sync: materialize recomputed old-policy comparison.
    assert float(max_error.item()) < 1e-6
    assert batch.mask.tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert model.cached_call_shapes == [(1, 2), (2, 2)]
