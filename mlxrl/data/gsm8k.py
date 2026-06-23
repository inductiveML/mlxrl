"""Tiny built-in GSM8K-style samples for Phase 1 sanity runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GSM8KExample:
    """A minimal GSM8K-style question and answer pair."""

    question: str
    answer: str


MINI_GSM8K: tuple[GSM8KExample, ...] = (
    GSM8KExample(
        question="Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        answer="Natalia sold 48/2 = 24 clips in May. She sold 48+24 = 72 clips. #### 72",
    ),
    GSM8KExample(
        question="Weng earns $12 an hour for babysitting. Yesterday, she babysat for 4 hours. "
        "How much did she earn?",
        answer="Weng earned 12*4 = 48 dollars. #### 48",
    ),
    GSM8KExample(
        question="Betty has 24 apples. She gives 9 to her neighbor and buys 6 more. "
        "How many apples does Betty have now?",
        answer="Betty has 24-9+6 = 21 apples. #### 21",
    ),
    GSM8KExample(
        question="A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        answer="The white fiber is 2/2 = 1 bolt, so total fiber is 2+1 = 3. #### 3",
    ),
)


def format_gsm8k_prompt(example: GSM8KExample) -> str:
    """Format a GSM8K prompt with the required answer tag."""

    return (
        "Solve the grade-school math problem. Show concise reasoning, then put only "
        "the final numeric answer in <answer>...</answer>.\n\n"
        f"Question: {example.question}"
    )


def format_gsm8k_answer_only_prompt(example: GSM8KExample) -> str:
    """Format a short Phase 1 sanity prompt that reaches reward quickly."""

    return (
        "Return exactly one XML tag containing the final numeric answer.\n"
        "Example:\n"
        "Question: Tom has 2 marbles and buys 3 more. How many marbles does he have?\n"
        "Answer: <answer>5</answer>\n\n"
        f"Question: {example.question}\n"
        "Answer:"
    )
