from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from mlxrl.algo.gigpo import GiGPOAlgorithm
from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.rollout.naive import Completion
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step
from mlxrl.train.trajectory import batch_from_trajectories, optimizer_step_trajectory
from mlxrl.trajectory import trajectory_from_single_turn

pytestmark = pytest.mark.metal


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.proj(self.embedding(tokens))


def _model(seed: int = 21) -> TinyPolicy:
    mx.random.seed(seed)
    return TinyPolicy()


def _single_turn_data() -> tuple[tuple[Completion, ...], tuple[float, ...]]:
    completions = (
        Completion(0, 0, (1, 2), (3, 4), (), "a"),
        Completion(0, 1, (1, 2), (5,), (), "b"),
    )
    rewards = (1.0, 3.0)
    return completions, rewards


def test_gigpo_omega_zero_single_turn_matches_grpo_loss_and_gradient() -> None:
    completions, rewards = _single_turn_data()
    trajectories = tuple(
        trajectory_from_single_turn(
            task_index=completion.prompt_index,
            group_index=completion.group_index,
            task="prompt",
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            completion_text=completion.text,
            reward=reward,
            state_id=("prompt", completion.prompt_index),
        )
        for completion, reward in zip(completions, rewards, strict=True)
    )

    grpo_model = _model()
    gigpo_model = _model()
    grpo_optimizer = optim.SGD(learning_rate=0.01)
    gigpo_optimizer = optim.SGD(learning_rate=0.01)
    grpo_batch = batch_from_rollouts(
        grpo_model,
        completions,
        rewards,
        group_size=2,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
        compute_reference=False,
    )
    gigpo_batch = batch_from_trajectories(
        gigpo_model,
        trajectories,
        group_size=2,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
        compute_reference=False,
    )

    grpo_metrics = optimizer_step(
        grpo_model,
        grpo_optimizer,
        grpo_batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GRPOAlgorithm(),
    )
    gigpo_metrics = optimizer_step_trajectory(
        gigpo_model,
        gigpo_optimizer,
        gigpo_batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=GiGPOAlgorithm(omega=0.0),
    )

    assert abs(grpo_metrics.loss - gigpo_metrics.loss) < 1e-6
    assert abs(grpo_metrics.policy_gradient_loss - gigpo_metrics.policy_gradient_loss) < 1e-6
    assert abs(grpo_metrics.kl - gigpo_metrics.kl) < 1e-6
    assert abs(grpo_metrics.mean_ratio - gigpo_metrics.mean_ratio) < 1e-6
    assert grpo_metrics.mean_reward == gigpo_metrics.mean_reward

    errors = []
    for (grpo_path, grpo_param), (gigpo_path, gigpo_param) in zip(
        tree_flatten(grpo_model.trainable_parameters()),
        tree_flatten(gigpo_model.trainable_parameters()),
        strict=True,
    ):
        assert grpo_path == gigpo_path
        grpo_array = cast(Any, grpo_param)
        gigpo_array = cast(Any, gigpo_param)
        errors.append(mx.max(mx.abs(grpo_array - gigpo_array)))
    mx.eval(*errors)  # Test sync: materialize post-update GRPO/GiGPO parameter deltas.
    assert max(float(error.item()) for error in errors) < 1e-6


def test_optimizer_step_trajectory_micro_batch_matches_whole_batch_token_mean() -> None:
    completions = (
        Completion(0, 0, (1, 2), (3, 4), (), "a"),
        Completion(0, 1, (1, 2), (5,), (), "b"),
        Completion(1, 0, (6, 7), (8, 9, 10), (), "c"),
        Completion(1, 1, (6, 7), (11, 12), (), "d"),
    )
    rewards = (1.0, 0.0, 0.5, -0.5)
    trajectories = tuple(
        trajectory_from_single_turn(
            task_index=completion.prompt_index,
            group_index=completion.group_index,
            task="prompt",
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            completion_text=completion.text,
            reward=reward,
            state_id=("prompt", completion.prompt_index),
        )
        for completion, reward in zip(completions, rewards, strict=True)
    )
    algorithm = GiGPOAlgorithm(omega=0.0)
    whole_model = _model()
    micro_model = _model()
    whole_optimizer = optim.SGD(learning_rate=0.01)
    micro_optimizer = optim.SGD(learning_rate=0.01)
    whole_batch = batch_from_trajectories(
        whole_model,
        trajectories,
        group_size=2,
        pad_token_id=0,
        algorithm=algorithm,
        compute_reference=False,
    )
    micro_batch = batch_from_trajectories(
        micro_model,
        trajectories,
        group_size=2,
        pad_token_id=0,
        algorithm=algorithm,
        compute_reference=False,
    )

    whole = optimizer_step_trajectory(
        whole_model,
        whole_optimizer,
        whole_batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=algorithm,
        micro_batch_size=0,
    )
    micro = optimizer_step_trajectory(
        micro_model,
        micro_optimizer,
        micro_batch,
        beta=0.04,
        pad_token_id=0,
        algorithm=algorithm,
        micro_batch_size=2,
    )

    assert abs(whole.loss - micro.loss) < 1e-6
    assert abs(whole.policy_gradient_loss - micro.policy_gradient_loss) < 1e-6
    assert abs(whole.kl - micro.kl) < 1e-6
    assert abs(whole.mean_ratio - micro.mean_ratio) < 1e-6
    assert abs(whole.clip_fraction - micro.clip_fraction) < 1e-6
    assert whole.mean_reward == micro.mean_reward

    errors = []
    for (whole_path, whole_param), (micro_path, micro_param) in zip(
        tree_flatten(whole_model.trainable_parameters()),
        tree_flatten(micro_model.trainable_parameters()),
        strict=True,
    ):
        assert whole_path == micro_path
        whole_array = cast(Any, whole_param)
        micro_array = cast(Any, micro_param)
        errors.append(mx.max(mx.abs(whole_array - micro_array)))
    mx.eval(*errors)  # Test sync: materialize post-update microbatch parameter deltas.
    assert max(float(error.item()) for error in errors) < 1e-6
