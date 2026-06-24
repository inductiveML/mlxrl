import mlx.core as mx
import pytest

from mlxrl.algo.gigpo import (
    GiGPOAlgorithm,
    anchor_state_groups,
    anchor_state_step_advantages,
    compute_gigpo_step_advantages,
    flatten_step_records,
)
from mlxrl.trajectory import ActionSpan, Trajectory, TrajectoryStep

pytestmark = pytest.mark.metal


def _step(
    state_id: str,
    reward: float,
    token: int,
    done: bool = False,
) -> TrajectoryStep:
    return TrajectoryStep(
        observation=state_id,
        state_id=state_id,
        action_text=f"a{token}",
        action_tokens=(token,),
        old_policy_logprobs=(0.0,),
        reward=reward,
        done=done,
    )


def _trajectory(
    group_index: int,
    steps: tuple[TrajectoryStep, ...],
) -> Trajectory:
    full_tokens = [1]
    spans: list[ActionSpan] = []
    for index, step in enumerate(steps):
        start = len(full_tokens)
        full_tokens.extend(step.action_tokens)
        spans.append(ActionSpan(step_index=index, start=start, end=len(full_tokens)))
        if index != len(steps) - 1:
            full_tokens.append(10 + index)
    return Trajectory(
        task_index=0,
        group_index=group_index,
        task="task",
        initial_observation=steps[0].observation,
        full_token_ids=tuple(full_tokens),
        action_spans=tuple(spans),
        steps=steps,
        done=steps[-1].done,
    )


def test_return_to_go_and_anchor_groups_match_hand_example() -> None:
    trajectories = (
        _trajectory(0, (_step("A", 0.0, 2), _step("B", 2.0, 3, done=True))),
        _trajectory(1, (_step("A", 1.0, 4), _step("C", 0.0, 5, done=True))),
    )

    records = flatten_step_records(trajectories, gamma=1.0)
    groups = anchor_state_groups(records)
    micro = anchor_state_step_advantages(records, normalization="std")

    assert [record.return_to_go for record in records] == [2.0, 2.0, 1.0, 0.0]
    assert groups["A"] == (0, 2)
    assert groups["B"] == (1,)
    assert groups["C"] == (3,)
    assert micro[1] == 0.0
    assert micro[3] == 0.0
    assert abs(micro[0] - 1.0) < 1e-6
    assert abs(micro[2] + 1.0) < 1e-6


def test_gigpo_combined_advantages_match_hand_example() -> None:
    trajectories = (
        _trajectory(0, (_step("A", 0.0, 2), _step("B", 2.0, 3, done=True))),
        _trajectory(1, (_step("A", 1.0, 4), _step("C", 0.0, 5, done=True))),
    )

    advantages = compute_gigpo_step_advantages(
        trajectories=trajectories,
        group_size=2,
        omega=0.5,
        gamma=1.0,
        normalization="std",
    )

    expected = (1.5, 1.0, -1.5, -1.0)
    for actual, target in zip(advantages, expected, strict=True):
        assert abs(actual - target) < 1e-6


def test_gigpo_rejects_invalid_normalization() -> None:
    algorithm = GiGPOAlgorithm(normalization="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="normalization"):
        algorithm.compute_step_advantages(
            (
                _trajectory(
                    0,
                    (_step("A", 0.0, 2, done=True),),
                ),
            ),
            group_structure=1,
        )


def test_gigpo_multiturn_loss_and_gradient_match_hand_computation() -> None:
    algorithm = GiGPOAlgorithm(omega=1.0)
    old_policy = mx.zeros((2, 2), dtype=mx.float32)
    policy = mx.log(mx.array([[1.1, 0.9], [1.2, 0.8]], dtype=mx.float32))
    advantages = mx.array([[1.5, 1.0], [-1.5, -1.0]], dtype=mx.float32)
    mask = mx.ones_like(policy)

    def loss_fn(policy_logprobs: mx.array) -> mx.array:
        return algorithm.compute_loss(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=old_policy,
            reference_logprobs=policy_logprobs,
            advantages=advantages,
            action_mask=mask,
            beta=0.0,
        ).loss

    loss = loss_fn(policy)
    gradient = mx.grad(loss_fn)(policy)
    ratio = mx.exp(policy - old_policy)
    expected_loss = mx.mean(-ratio * advantages)
    expected_gradient = -ratio * advantages / 4.0
    max_gradient_error = mx.max(mx.abs(gradient - expected_gradient))
    mx.eval(  # Test sync: materialize GiGPO toy multi-turn loss and gradient.
        loss,
        expected_loss,
        max_gradient_error,
    )

    assert abs(float(loss.item()) - float(expected_loss.item())) < 1e-6
    assert float(max_gradient_error.item()) < 1e-6
