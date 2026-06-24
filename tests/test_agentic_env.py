from __future__ import annotations

from mlxrl.env import (
    RecurringStateTextEnv,
    SingleTurnRewardEnv,
    coerce_step_result,
    default_action_parser,
)
from mlxrl.trajectory import ActionSpan, Trajectory, TrajectoryStep, trajectory_from_single_turn


def test_reference_env_resets_and_steps_deterministically() -> None:
    env = RecurringStateTextEnv(max_turns=3)

    observation = env.reset()
    first = env.step("loop")
    second = env.step("advance")
    third = env.step("finish")

    assert observation == "task=reach goal; state=start"
    assert first.observation == "task=reach goal; state=start"
    assert first.reward == 0.0
    assert first.done is False
    assert second.observation == "task=reach goal; state=middle"
    assert second.reward == 0.25
    assert third.observation == "done"
    assert third.reward == 1.0
    assert third.done is True


def test_single_turn_env_wraps_reward_function() -> None:
    env = SingleTurnRewardEnv(prompt="2 + 2?", reward_fn=lambda text: 1.0 if "4" in text else 0.0)

    observation = env.reset()
    result = env.step("<answer>4</answer>")

    assert observation == "2 + 2?"
    assert result.reward == 1.0
    assert result.done is True
    assert env.state_id(observation) == ("single-turn", "2 + 2?")


def test_default_action_parser_prefers_action_tags() -> None:
    assert default_action_parser("think\n<action> finish </action>") == "finish"
    assert default_action_parser("  finish  ") == "finish"


def test_coerce_step_result_accepts_tuple_results() -> None:
    result = coerce_step_result(("obs", 0.5, False, {"x": 1}))

    assert result.observation == "obs"
    assert result.reward == 0.5
    assert result.done is False
    assert result.info == {"x": 1}


def test_single_turn_round_trips_through_trajectory_structures() -> None:
    trajectory = trajectory_from_single_turn(
        task_index=0,
        group_index=1,
        task="prompt",
        prompt_tokens=(10, 11),
        completion_tokens=(12, 13),
        completion_text="answer",
        reward=2.0,
        state_id="prompt-state",
    )

    assert trajectory.full_token_ids == (10, 11, 12, 13)
    assert trajectory.action_token_ids() == (12, 13)
    assert trajectory.total_return == 2.0
    assert trajectory.discounted_returns_to_go(gamma=0.5) == (2.0,)
    assert trajectory.action_spans == (ActionSpan(step_index=0, start=2, end=4),)
    assert trajectory.steps[0].state_id == "prompt-state"


def test_trajectory_validates_action_span_alignment() -> None:
    step = TrajectoryStep(
        observation="obs",
        state_id="s",
        action_text="a",
        action_tokens=(3,),
        old_policy_logprobs=(0.0,),
        reward=0.0,
        done=True,
    )
    trajectory = Trajectory(
        task_index=0,
        group_index=0,
        task="task",
        initial_observation="obs",
        full_token_ids=(1, 2, 3),
        action_spans=(ActionSpan(step_index=0, start=2, end=3),),
        steps=(step,),
        done=True,
    )

    assert trajectory.action_token_count == 1
