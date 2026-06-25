from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

import mlxrl.algo.grpo as grpo_module
from mlxrl.algo.grpo import DAPOAlgorithm, GRPOAlgorithm, GSPOAlgorithm
from mlxrl.policy.logprobs import prepare_completion_logprob_inputs
from mlxrl.rollout.naive import Completion
from mlxrl.train.grpo import (
    GRPOBatch,
    batch_from_rollouts,
    grpo_metrics_from_batch,
    optimizer_step,
)

pytestmark = pytest.mark.metal


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.proj(self.embedding(tokens))


class ModeRecordingPolicy(TinyPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.call_training_modes: list[bool] = []

    def __call__(self, tokens: mx.array) -> mx.array:
        self.call_training_modes.append(bool(self.training))
        return super().__call__(tokens)


def _model(seed: int = 11) -> TinyPolicy:
    mx.random.seed(seed)
    return TinyPolicy()


def _batch() -> GRPOBatch:
    prompt_token_ids = ((1, 2), (1, 2), (1, 2), (1, 2))
    completion_token_ids = ((3, 4), (5,), (6, 7, 8), (9, 10))
    return GRPOBatch(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        rewards=mx.array([1.0, 0.0, 0.5, -0.5], dtype=mx.float32),
        advantages=mx.array([1.0, -1.0, 0.5, -0.5], dtype=mx.float32),
        old_policy_logprobs=mx.zeros((4, 3), dtype=mx.float32),
        reference_logprobs=mx.zeros((4, 3), dtype=mx.float32),
        mask=mx.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=mx.float32,
        ),
    )


def test_optimizer_step_micro_batch_matches_whole_batch_token_mean() -> None:
    whole_model = _model()
    micro_model = _model()
    whole_optimizer = optim.SGD(learning_rate=0.01)
    micro_optimizer = optim.SGD(learning_rate=0.01)
    batch = _batch()

    whole = optimizer_step(
        whole_model,
        whole_optimizer,
        batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
        micro_batch_size=0,
    )
    micro = optimizer_step(
        micro_model,
        micro_optimizer,
        batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
        micro_batch_size=2,
    )

    assert abs(whole.loss - micro.loss) < 1e-6
    assert abs(whole.policy_gradient_loss - micro.policy_gradient_loss) < 1e-6
    assert abs(whole.kl - micro.kl) < 1e-6
    assert abs(whole.mean_ratio - micro.mean_ratio) < 1e-6
    assert abs(whole.clip_fraction - micro.clip_fraction) < 1e-6
    assert whole.mean_reward == micro.mean_reward

    param_errors = []
    for (whole_path, whole_param), (micro_path, micro_param) in zip(
        tree_flatten(whole_model.trainable_parameters()),
        tree_flatten(micro_model.trainable_parameters()),
        strict=True,
    ):
        assert whole_path == micro_path
        whole_array = cast(Any, whole_param)
        micro_array = cast(Any, micro_param)
        param_errors.append(mx.max(mx.abs(whole_array - micro_array)))
    mx.eval(*param_errors)  # Test sync: materialize post-update parameter comparison.
    assert max(float(error.item()) for error in param_errors) < 1e-6


def test_grpo_metrics_skips_kl_for_dummy_reference_beta_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    batch = replace(_batch(), reference_is_policy=True)

    def fail_kl(policy_logprobs: mx.array, reference_logprobs: mx.array) -> mx.array:
        del policy_logprobs, reference_logprobs
        raise AssertionError("dummy-reference beta-zero path should not compute KL")

    monkeypatch.setattr(grpo_module, "approximate_kl", fail_kl)

    metrics = grpo_metrics_from_batch(
        model,
        batch,
        beta=0.0,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
    )
    mx.eval(metrics.loss, metrics.kl)  # Test sync: materialize beta-zero fast path.

    assert float(metrics.kl.item()) == 0.0


def test_grpo_metrics_rejects_dummy_reference_with_nonzero_beta() -> None:
    model = _model()
    batch = replace(_batch(), reference_is_policy=True)

    with pytest.raises(ValueError, match=r"Set beta=0\.0 or build the batch"):
        grpo_metrics_from_batch(
            model,
            batch,
            beta=0.04,
            pad_token_id=0,
            algorithm=GRPOAlgorithm(),
        )


def test_batch_from_rollouts_uses_eval_mode_and_restores_train_mode() -> None:
    model = ModeRecordingPolicy()
    model.train()
    completions = (
        Completion(0, 0, (1, 2), (3, 4), (), "a"),
        Completion(0, 1, (1, 2), (5,), (), "b"),
    )

    batch = batch_from_rollouts(
        model,
        completions,
        rewards=(1.0, 0.0),
        group_size=2,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
        compute_reference=False,
    )

    assert model.call_training_modes == [False]
    assert model.training is True
    mx.eval(  # Test sync: materialize batch logprobs after mode restoration.
        batch.old_policy_logprobs,
        batch.mask,
    )


def test_grpo_metrics_accepts_prepared_logprob_inputs() -> None:
    model = _model()
    batch = _batch()
    prepared_batch = replace(
        batch,
        logprob_inputs=prepare_completion_logprob_inputs(
            batch.prompt_token_ids,
            batch.completion_token_ids,
            pad_token_id=0,
        ),
    )

    direct = grpo_metrics_from_batch(
        model,
        batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
    )
    prepared = grpo_metrics_from_batch(
        model,
        prepared_batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
    )
    errors = (
        mx.abs(direct.loss - prepared.loss),
        mx.abs(direct.policy_gradient_loss - prepared.policy_gradient_loss),
        mx.abs(direct.kl - prepared.kl),
        mx.abs(direct.mean_ratio - prepared.mean_ratio),
    )
    mx.eval(*errors)  # Test sync: materialize prepared/direct metrics comparison.

    assert max(float(error.item()) for error in errors) < 1e-6


def test_optimizer_step_rejects_micro_batching_for_sequence_reductions() -> None:
    model = _model()
    optimizer = optim.SGD(learning_rate=0.01)

    with pytest.raises(ValueError, match="token-mean policy losses only"):
        optimizer_step(
            model,
            optimizer,
            _batch(),
            beta=0.04,
            pad_token_id=0,
            algorithm=GSPOAlgorithm(),
            micro_batch_size=2,
        )


def test_dapo_filter_batch_drops_zero_advantage_groups() -> None:
    batch = GRPOBatch(
        prompt_token_ids=((1,), (1,), (2,), (2,)),
        completion_token_ids=((3,), (4,), (5,), (6,)),
        rewards=mx.array([0.0, 0.0, 1.0, 0.0], dtype=mx.float32),
        advantages=mx.array([0.0, 0.0, 1.0, -1.0], dtype=mx.float32),
        old_policy_logprobs=mx.zeros((4, 1), dtype=mx.float32),
        reference_logprobs=mx.zeros((4, 1), dtype=mx.float32),
        mask=mx.ones((4, 1), dtype=mx.float32),
        reference_is_policy=True,
        logprob_inputs=prepare_completion_logprob_inputs(
            ((1,), (1,), (2,), (2,)),
            ((3,), (4,), (5,), (6,)),
            pad_token_id=0,
        ),
    )

    filtered = DAPOAlgorithm(dynamic_sampling=True).filter_batch(
        batch,
        group_structure=2,
    )
    mx.eval(  # Test sync: materialize DAPO filtered batch arrays.
        filtered.rewards,
        filtered.advantages,
        filtered.mask,
    )

    assert filtered.prompt_token_ids == ((2,), (2,))
    assert filtered.completion_token_ids == ((5,), (6,))
    assert filtered.rewards.tolist() == [1.0, 0.0]
    assert filtered.advantages.tolist() == [1.0, -1.0]
    assert filtered.reference_is_policy is True
    assert filtered.logprob_inputs is not None
    assert filtered.logprob_inputs.input_ids.tolist() == [[2], [2]]
