from __future__ import annotations

from collections.abc import Sequence

import pytest

from mlxrl.env import RecurringStateTextEnv, SingleTurnRewardEnv, StepResult
from mlxrl.rollout.agentic import GeneratedAction, generate_agentic_trajectories
from mlxrl.rollout.naive import Completion, SamplingConfig

pytestmark = pytest.mark.metal


class ToyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) % 251 + 1 for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr((token - 1) % 251) for token in tokens)


def _scripted_action(
    task_index: int,
    group_index: int,
    turn_index: int,
    context_tokens: Sequence[int],
    observation: str,
) -> GeneratedAction:
    del task_index, context_tokens, observation
    if group_index == 0 and turn_index == 0:
        return GeneratedAction(tokens=(101,), old_policy_logprobs=(-0.1,), text="loop")
    return GeneratedAction(tokens=(102,), old_policy_logprobs=(-0.2,), text="finish")


def test_agentic_rollout_records_multiturn_trajectory_structure() -> None:
    trajectories = generate_agentic_trajectories(
        model=None,
        tokenizer=ToyTokenizer(),
        env_factory=lambda task, seed, group: RecurringStateTextEnv(
            task=str(task),
            max_turns=3,
        ),
        tasks=("task-a",),
        group_size=2,
        sampling=SamplingConfig(max_tokens=1),
        rollout_mode="parallel_per_turn",
        action_generator=_scripted_action,
    )

    assert len(trajectories) == 2
    first, second = trajectories
    assert len(first.steps) == 2
    assert first.steps[0].state_id == "start"
    assert first.steps[0].action_text == "loop"
    assert first.steps[0].reward == 0.0
    assert first.steps[1].action_text == "finish"
    assert first.steps[1].reward == 1.0
    assert first.done is True
    assert first.truncated is False
    assert first.action_token_ids() == (101, 102)
    assert second.steps[0].action_text == "finish"
    assert second.total_return == 1.0


def test_agentic_rollout_single_turn_matches_completion_shape() -> None:
    completion = Completion(
        prompt_index=0,
        group_index=0,
        prompt_tokens=(1, 2, 3),
        completion_tokens=(4, 5),
        old_policy_logprobs=(-0.4, -0.5),
        text="answer",
    )

    trajectory = generate_agentic_trajectories(
        model=None,
        tokenizer=ToyTokenizer(),
        env_factory=lambda task, seed, group: SingleTurnRewardEnv(
            prompt=str(task),
            reward_fn=lambda text: 1.0 if text == "answer" else 0.0,
        ),
        tasks=("prompt",),
        group_size=1,
        sampling=SamplingConfig(max_tokens=2),
        action_generator=lambda *_: GeneratedAction(
            tokens=completion.completion_tokens,
            old_policy_logprobs=completion.old_policy_logprobs,
            text=completion.text,
        ),
    )[0]

    assert trajectory.group_index == completion.group_index
    assert trajectory.steps[0].action_tokens == completion.completion_tokens
    assert trajectory.steps[0].old_policy_logprobs == completion.old_policy_logprobs
    assert trajectory.action_token_ids() == completion.completion_tokens
    assert trajectory.steps[0].reward == 1.0
    assert len(trajectory.steps) == 1


class SharedBatchEnv:
    max_turns = 1

    def __init__(self) -> None:
        self.batch_calls = 0

    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> StepResult:
        raise AssertionError(f"sequential step should not be used for {action}")

    def step_batch(self, actions: list[str]) -> list[StepResult]:
        self.batch_calls += 1
        return [
            StepResult(observation="done", reward=float(index), done=True)
            for index, _ in enumerate(actions)
        ]

    def state_id(self, observation: str) -> str:
        return observation


def test_agentic_rollout_uses_optional_batch_step_hook() -> None:
    env = SharedBatchEnv()
    trajectories = generate_agentic_trajectories(
        model=None,
        tokenizer=ToyTokenizer(),
        env_factory=lambda task, seed, group: env,
        tasks=("task",),
        group_size=3,
        sampling=SamplingConfig(max_tokens=1),
        action_generator=lambda *_: GeneratedAction(
            tokens=(7,),
            old_policy_logprobs=(-0.7,),
            text="finish",
        ),
    )

    assert env.batch_calls == 1
    assert [trajectory.total_return for trajectory in trajectories] == [0.0, 1.0, 2.0]
