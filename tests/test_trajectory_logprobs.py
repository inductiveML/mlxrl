from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxrl.policy.trajectory_logprobs import trajectory_action_logprobs
from mlxrl.trajectory import ActionSpan, Trajectory, TrajectoryStep

pytestmark = pytest.mark.metal


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.proj(self.embedding(tokens))


def _trajectory() -> Trajectory:
    return Trajectory(
        task_index=0,
        group_index=0,
        task="task",
        initial_observation="obs",
        full_token_ids=(1, 2, 3, 4, 5, 6),
        action_spans=(
            ActionSpan(step_index=0, start=2, end=4),
            ActionSpan(step_index=1, start=5, end=6),
        ),
        steps=(
            TrajectoryStep(
                observation="obs",
                state_id="s0",
                action_text="a",
                action_tokens=(3, 4),
                old_policy_logprobs=(0.0, 0.0),
                reward=0.0,
                done=False,
            ),
            TrajectoryStep(
                observation="obs2",
                state_id="s1",
                action_text="b",
                action_tokens=(6,),
                old_policy_logprobs=(0.0,),
                reward=1.0,
                done=True,
            ),
        ),
        done=True,
    )


def test_trajectory_logprobs_gather_action_positions_only() -> None:
    mx.random.seed(3)
    model = TinyPolicy()
    trajectory = _trajectory()

    gathered = trajectory_action_logprobs(model, [trajectory], pad_token_id=0)
    sequence = mx.array([trajectory.full_token_ids[:-1]], dtype=mx.int32)
    targets = mx.array([trajectory.full_token_ids[1:]], dtype=mx.int32)
    logits = model(sequence)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target_logprobs = mx.squeeze(
        mx.take_along_axis(logprobs, targets[..., None], axis=-1),
        axis=-1,
    )
    expected = mx.array(
        [
            [
                target_logprobs[0, 1].item(),
                target_logprobs[0, 2].item(),
                target_logprobs[0, 4].item(),
            ]
        ],
        dtype=mx.float32,
    )
    error = mx.max(mx.abs(gathered.logprobs - expected))
    mx.eval(  # Test sync: materialize trajectory logprob gather comparison.
        error,
        gathered.mask,
    )

    assert float(error.item()) < 1e-6
    assert gathered.mask.tolist() == [[1.0, 1.0, 1.0]]
