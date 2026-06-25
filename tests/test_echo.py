from __future__ import annotations

import math
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from mlxrl.algo.gigpo import GiGPOAlgorithm
from mlxrl.echo import ACTION, ECHO, MASKED, EchoSchedule
from mlxrl.train.trajectory import (
    batch_from_trajectories,
    optimizer_step_trajectory,
    trajectory_metrics_from_batch,
)
from mlxrl.trajectory import ActionSpan, Trajectory, TrajectoryStep

pytestmark = pytest.mark.metal


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.proj(self.embedding(tokens))


class ConstantLogitPolicy(nn.Module):
    vocab_size = 6

    def __init__(self) -> None:
        super().__init__()
        self.logits = mx.zeros((self.vocab_size,), dtype=mx.float32)

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size, sequence_length = tokens.shape
        return mx.broadcast_to(self.logits, (batch_size, sequence_length, self.vocab_size))


class SplitLogitPolicy(nn.Module):
    vocab_size = 6

    def __init__(self) -> None:
        super().__init__()
        self.action_logits = mx.zeros((self.vocab_size,), dtype=mx.float32)
        self.echo_logits = mx.zeros((self.vocab_size,), dtype=mx.float32)

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size, sequence_length = tokens.shape
        action = mx.broadcast_to(
            self.action_logits,
            (batch_size, sequence_length, self.vocab_size),
        )
        echo = mx.broadcast_to(
            self.echo_logits,
            (batch_size, sequence_length, self.vocab_size),
        )
        return mx.where(tokens[..., None] == 10, action, echo)


def _model(seed: int = 31) -> TinyPolicy:
    mx.random.seed(seed)
    return TinyPolicy()


def _trajectory(
    *,
    full_token_ids: tuple[int, ...],
    token_roles: tuple[int, ...] | None = None,
    token_advantages: tuple[float, ...] | None = None,
    action_start: int = 1,
    action_end: int = 2,
    reward: float = 1.0,
) -> Trajectory:
    action_tokens = full_token_ids[action_start:action_end]
    return Trajectory(
        task_index=0,
        group_index=0,
        task="echo",
        initial_observation="echo",
        full_token_ids=full_token_ids,
        action_spans=(ActionSpan(step_index=0, start=action_start, end=action_end),),
        steps=(
            TrajectoryStep(
                observation="echo",
                state_id="echo",
                action_text="a",
                action_tokens=action_tokens,
                old_policy_logprobs=tuple(0.0 for _ in action_tokens),
                reward=reward,
                done=True,
            ),
        ),
        done=True,
        token_roles=token_roles,
        token_advantages=token_advantages,
    )


def _batch(model: nn.Module, trajectory: Trajectory):
    return batch_from_trajectories(
        model,
        (trajectory,),
        group_size=1,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
        compute_reference=False,
    )


def _tree_array(params: Any, suffix: str) -> mx.array:
    matches = [
        cast(mx.array, value)
        for path, value in tree_flatten(params)
        if str(path).endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one parameter ending in {suffix}, got {len(matches)}")
    return matches[0]


def test_echo_alpha_zero_matches_current_gigpo_loss_and_gradient() -> None:
    trajectory = _trajectory(full_token_ids=(1, 2, 3), action_start=1, action_end=3)
    baseline_model = _model()
    echo_zero_model = _model()
    baseline_optimizer = optim.SGD(learning_rate=0.01)
    echo_zero_optimizer = optim.SGD(learning_rate=0.01)
    baseline_batch = _batch(baseline_model, trajectory)
    echo_zero_batch = _batch(echo_zero_model, trajectory)

    baseline = optimizer_step_trajectory(
        baseline_model,
        baseline_optimizer,
        baseline_batch,
        beta=0.0,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
    )
    echo_zero = optimizer_step_trajectory(
        echo_zero_model,
        echo_zero_optimizer,
        echo_zero_batch,
        beta=0.0,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
        echo_alpha=0.0,
    )

    assert baseline.loss == echo_zero.loss
    assert baseline.policy_gradient_loss == echo_zero.policy_gradient_loss
    assert baseline.kl == echo_zero.kl

    errors = []
    for (left_path, left_param), (right_path, right_param) in zip(
        tree_flatten(baseline_model.trainable_parameters()),
        tree_flatten(echo_zero_model.trainable_parameters()),
        strict=True,
    ):
        assert left_path == right_path
        errors.append(mx.max(mx.abs(cast(Any, left_param) - cast(Any, right_param))))
    mx.eval(*errors)  # Test sync: materialize alpha-zero parameter deltas.
    assert max(float(error.item()) for error in errors) == 0.0


def test_echo_learns_fixed_echo_targets() -> None:
    model = ConstantLogitPolicy()
    optimizer = optim.SGD(learning_rate=0.5)
    trajectory = _trajectory(
        full_token_ids=(0, 3, 3, 3),
        token_roles=(MASKED, ECHO, ECHO, ECHO),
        action_start=1,
        action_end=2,
    )
    batch = _batch(model, trajectory)
    before = trajectory_metrics_from_batch(
        model,
        batch,
        beta=0.0,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
        echo_alpha=1.0,
    )
    mx.eval(before.loss_echo, before.echo_accuracy)  # Test sync: read initial ECHO metrics.
    assert before.loss_echo is not None
    assert before.echo_accuracy is not None

    after = None
    for _ in range(25):
        after = optimizer_step_trajectory(
            model,
            optimizer,
            batch,
            beta=0.0,
            pad_token_id=0,
            algorithm=GiGPOAlgorithm(omega=0.0),
            echo_alpha=1.0,
        )
    assert after is not None

    assert after.loss_echo < float(before.loss_echo.item())
    assert after.echo_accuracy > float(before.echo_accuracy.item())


def test_echo_normalization_keeps_action_gradient_stable_when_echo_is_duplicated() -> None:
    short = _trajectory(
        full_token_ids=(10, 1, 2),
        token_roles=(MASKED, ACTION, ECHO),
        token_advantages=(0.0, 1.0, 0.0),
        action_start=1,
        action_end=2,
    )
    long = _trajectory(
        full_token_ids=(10, 1, 2, 2, 2),
        token_roles=(MASKED, ACTION, ECHO, ECHO, ECHO),
        token_advantages=(0.0, 1.0, 0.0, 0.0, 0.0),
        action_start=1,
        action_end=2,
    )
    short_model = SplitLogitPolicy()
    long_model = SplitLogitPolicy()
    short_batch = _batch(short_model, short)
    long_batch = _batch(long_model, long)

    def loss_fn(model: nn.Module, batch: Any) -> mx.array:
        return trajectory_metrics_from_batch(
            model,
            batch,
            beta=0.0,
            pad_token_id=0,
            algorithm=GiGPOAlgorithm(omega=0.0),
            echo_alpha=1.0,
        ).loss

    _, short_grad = nn.value_and_grad(short_model, lambda m: loss_fn(m, short_batch))(
        short_model
    )
    _, long_grad = nn.value_and_grad(long_model, lambda m: loss_fn(m, long_batch))(
        long_model
    )
    action_error = mx.max(
        mx.abs(
            _tree_array(short_grad, "action_logits")
            - _tree_array(long_grad, "action_logits")
        )
    )
    mx.eval(action_error)  # Test sync: materialize duplicated-ECHO action grad delta.

    assert float(action_error.item()) < 1e-6


@pytest.mark.parametrize(
    ("roles", "advantages"),
    [
        ((MASKED, MASKED, MASKED), (0.0, 0.0, 0.0)),
        ((MASKED, ACTION, ACTION), (0.0, 1.0, 1.0)),
        ((MASKED, ECHO, ECHO), (0.0, 0.0, 0.0)),
    ],
)
def test_echo_edge_role_batches_stay_finite(
    roles: tuple[int, ...],
    advantages: tuple[float, ...],
) -> None:
    model = SplitLogitPolicy()
    trajectory = _trajectory(
        full_token_ids=(10, 1, 2),
        token_roles=roles,
        token_advantages=advantages,
        action_start=1,
        action_end=3,
    )
    batch = _batch(model, trajectory)
    metrics = trajectory_metrics_from_batch(
        model,
        batch,
        beta=0.0,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
        echo_alpha=0.5,
    )
    mx.eval(metrics.loss, metrics.loss_echo, metrics.echo_accuracy)

    assert math.isfinite(float(metrics.loss.item()))
    assert metrics.loss_echo is not None
    assert math.isfinite(float(metrics.loss_echo.item()))
    assert metrics.echo_accuracy is not None
    assert math.isfinite(float(metrics.echo_accuracy.item()))


def test_masked_tokens_have_zero_gradient_even_with_echo_alpha() -> None:
    model = SplitLogitPolicy()
    trajectory = _trajectory(
        full_token_ids=(10, 1, 2),
        token_roles=(MASKED, ACTION, MASKED),
        token_advantages=(0.0, 1.0, 0.0),
        action_start=1,
        action_end=2,
    )
    batch = _batch(model, trajectory)

    def loss_fn(model: nn.Module) -> mx.array:
        return trajectory_metrics_from_batch(
            model,
            batch,
            beta=0.0,
            pad_token_id=0,
            algorithm=GiGPOAlgorithm(omega=0.0),
            echo_alpha=1.0,
        ).loss

    _, gradients = nn.value_and_grad(model, loss_fn)(model)
    masked_only_grad = mx.max(mx.abs(_tree_array(gradients, "echo_logits")))
    mx.eval(masked_only_grad)  # Test sync: materialize masked-token gradient.

    assert float(masked_only_grad.item()) == 0.0


def test_echo_schedule_linear_taper_to_zero() -> None:
    schedule = EchoSchedule(alpha=0.1, schedule="linear_taper_to_zero", taper_steps=4)

    assert schedule.value(0) == 0.1
    assert schedule.value(2) == 0.05
    assert schedule.value(4) == 0.0
    assert schedule.value(99) == 0.0
