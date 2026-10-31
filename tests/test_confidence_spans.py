from __future__ import annotations

import pytest

from mlxrl.data.confidence import arithmetic_steps


class CharacterTokenizer:
    def decode(self, ids):
        return "".join(chr(x) for x in ids)


def test_arithmetic_reader_scores_last_result_and_excludes_all_values():
    text = (
        "step: a = -2, b = a + 5 = 3\nstep: c = b * 2 = 6\n<answer>6</answer>\nstep: d = c + 1 = 7"
    )
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert len(steps) == 2
    assert text[steps[0].results[0].start : steps[0].results[0].end] == "3"
    assert [text[s.start : s.end] for s in steps[0].exclusions] == ["-2", "3"]
    assert steps[-1].span.end <= text.index("<answer>")


def test_unscored_formula_and_root_lines_are_neutral():
    text = "step: a = 2\nstep: b = a + 3\nstep: c = 3.5 - 1 = 2.5\n#### 2.5"
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert [len(step.results) for step in steps] == [0, 0, 1]
    assert text[steps[2].results[0].start : steps[2].results[0].end] == "2.5"


def test_scientific_notation_exponent_is_not_an_arithmetic_operator():
    text = "step: a = 2e-3\nstep: b = 1e+3"
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert len(steps) == 2 and all(step.results == () for step in steps)


def test_negative_root_is_unscored_and_inline_answer_is_excluded():
    text = "step: a = -2\nstep: b = a + 5 = 3 <answer>3</answer> step: c = b + 1 = 4"
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert len(steps) == 2
    assert steps[0].results == ()
    assert steps[1].span.end <= text.index("<answer>")


@pytest.mark.parametrize("text", [
    "step: a = 2, b = a + 3",
    "step: a = 2, b = a + 3 =",
    "step: a = 2, b = a + 3 = unfinished",
    "step: a = 2, b = a + 3 = 1, 000",
])
def test_unfinished_result_cannot_reuse_an_earlier_assignment(text):
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert len(steps) == 1 and steps[0].results == ()


@pytest.mark.parametrize("result", ["1,000", "-1,024.5", ".5", "2e3", "-3.2e-4"])
def test_complete_numeric_result_span_is_scored_and_excluded(result):
    text = f"step: x = 2 * 3 = {result}\n<answer>{result}</answer>"
    steps = arithmetic_steps(CharacterTokenizer(), tuple(map(ord, text)))
    assert len(steps) == 1
    span, = steps[0].results
    assert text[span.start:span.end] == result
    assert span in steps[0].exclusions


def test_token_that_contains_result_is_excluded_as_a_whole():
    class Tokenizer:
        tokens = ("step: a", " = 2 +", " 3", " = 5", "\n", "<answer>5</answer>")

        def decode(self, ids):
            return "".join(self.tokens[i] for i in ids)

    steps = arithmetic_steps(Tokenizer(), range(6))
    assert len(steps) == 1
    assert (steps[0].results[0].start, steps[0].results[0].end) == (3, 4)


def test_unstable_decoding_and_unstructured_prose_fail_closed():
    class Tokenizer:
        def decode(self, ids):
            return "bad prefix" if len(ids) == 1 else "step: a = 2 + 3 = 5"

    assert arithmetic_steps(Tokenizer(), [1, 2]) == ()
    assert arithmetic_steps(CharacterTokenizer(), list(map(ord, "The answer is 5."))) == ()
    assert arithmetic_steps(CharacterTokenizer(), []) == ()
