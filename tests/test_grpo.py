from __future__ import annotations

import math

import mlx.core as mx

from mlxrl.algo.grpo import (
    approximate_kl,
    group_normalize_rewards,
    grpo_loss,
)


def test_group_normalize_rewards_two_item_group() -> None:
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)

    advantages = group_normalize_rewards(rewards, group_size=2)
    mx.eval(advantages)  # Test sync: materialize values before Python list assertion.

    assert advantages.tolist() == [-1.0, 1.0]


def test_grpo_loss_matches_hand_computed_tiny_vocab_example() -> None:
    policy = mx.log(mx.array([[0.20, 0.50], [0.40, 0.25]], dtype=mx.float32))
    reference = mx.log(mx.array([[0.25, 0.40], [0.50, 0.20]], dtype=mx.float32))
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)
    advantages = group_normalize_rewards(rewards, group_size=2)
    beta = 0.1

    metrics = grpo_loss(
        policy_logprobs=policy,
        old_policy_logprobs=policy,
        reference_logprobs=reference,
        advantages=advantages,
        mask=mx.ones_like(policy),
        beta=beta,
    )
    mx.eval(metrics.loss, metrics.kl)  # Test sync: materialize scalars for assertions.

    manual_kl_values = []
    for p_row, r_row in [
        ([0.20, 0.50], [0.25, 0.40]),
        ([0.40, 0.25], [0.50, 0.20]),
    ]:
        for p_value, r_value in zip(p_row, r_row, strict=True):
            log_ratio = math.log(r_value) - math.log(p_value)
            manual_kl_values.append(math.exp(log_ratio) - log_ratio - 1.0)
    manual_kl = sum(manual_kl_values) / len(manual_kl_values)
    manual_pg = ((1.0 + 1.0) + (-1.0 + -1.0)) / 4
    expected_loss = manual_pg + beta * manual_kl

    assert abs(float(metrics.kl.item()) - manual_kl) < 1e-6
    assert abs(float(metrics.loss.item()) - expected_loss) < 1e-6


def test_approximate_kl_is_zero_when_policies_match() -> None:
    logprobs = mx.log(mx.array([[0.3, 0.7]], dtype=mx.float32))

    kl = approximate_kl(logprobs, logprobs)
    mx.eval(kl)  # Test sync: materialize values before Python list assertion.

    assert kl.tolist() == [[0.0, 0.0]]
