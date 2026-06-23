from __future__ import annotations

import pytest

from mlxrl.data.rewards import (
    accuracy_reward,
    extract_answer,
    format_reward,
    get_reward,
    list_rewards,
    reward,
)


def test_reward_registry_decorator_registers_function() -> None:
    name = "unit_test_constant"

    @reward(name)
    def constant_reward(**_: object) -> float:
        return 0.25

    assert name in list_rewards()
    assert get_reward(name)() == 0.25


def test_reward_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Reward already registered"):

        @reward("accuracy")
        def duplicate(**_: object) -> float:
            return 0.0


def test_extract_answer_uses_tag_then_gsm8k_then_last_number() -> None:
    assert extract_answer("work #### 42") == "42"
    assert extract_answer("work <answer>7</answer> #### 42") == "7"
    assert extract_answer("The result is 1,234.50 after checking.") == "1,234.50"
    assert extract_answer("no numeric answer") is None


def test_accuracy_reward_matches_normalized_extracted_answers() -> None:
    completion = "Reasoning here. <answer>1,234</answer>"
    answer = "The answer is #### 1234"

    assert accuracy_reward(completion, answer=answer) == 1.0
    assert accuracy_reward("<answer>48 + 24 = 72</answer>", answer="#### 72") == 1.0
    assert accuracy_reward("<answer>5</answer>", answer="#### 6") == 0.0
    assert accuracy_reward("no answer", answer="#### 6") == 0.0


def test_format_reward_requires_non_empty_answer_tag() -> None:
    assert format_reward("some work <answer>4</answer>") == 1.0
    assert format_reward("some work <answer> </answer>") == 0.0
    assert format_reward("some work #### 4") == 0.0
