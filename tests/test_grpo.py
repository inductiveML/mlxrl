from __future__ import annotations

import math

import mlx.core as mx

from mlxrl.algo.grpo import (
    DAPOAlgorithm,
    DrGRPOAlgorithm,
    GSPOAlgorithm,
    approximate_kl,
    group_center_rewards,
    group_normalize_rewards,
    grpo_loss,
)


def test_group_normalize_rewards_two_item_group() -> None:
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)

    advantages = group_normalize_rewards(rewards, group_size=2)
    mx.eval(advantages)  # Test sync: materialize values before Python list assertion.

    assert advantages.tolist() == [-1.0, 1.0]


def test_group_center_rewards_two_item_group() -> None:
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)

    advantages = group_center_rewards(rewards, group_size=2)
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


def test_phase3_variants_reduce_to_base_grpo_under_configs() -> None:
    policy = mx.log(mx.array([[0.20, 0.50], [0.40, 0.25]], dtype=mx.float32))
    old_policy = mx.log(mx.array([[0.18, 0.52], [0.43, 0.20]], dtype=mx.float32))
    reference = mx.log(mx.array([[0.25, 0.40], [0.50, 0.20]], dtype=mx.float32))
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)
    mask = mx.ones_like(policy)
    beta = 0.1
    base_advantages = group_normalize_rewards(rewards, group_size=2)
    base = grpo_loss(
        policy_logprobs=policy,
        old_policy_logprobs=old_policy,
        reference_logprobs=reference,
        advantages=base_advantages,
        mask=mask,
        beta=beta,
    )

    variants = [
        DAPOAlgorithm(clip_low=None, clip_high=None),
        DrGRPOAlgorithm(normalize_rewards=True, loss_reduction="token_mean"),
        GSPOAlgorithm(importance="token", clip_low=None, clip_high=None),
    ]
    losses = []
    for algorithm in variants:
        advantages = algorithm.advantages(rewards, group_size=2)
        metrics = algorithm.loss(
            policy_logprobs=policy,
            old_policy_logprobs=old_policy,
            reference_logprobs=reference,
            advantages=advantages,
            mask=mask,
            beta=beta,
        )
        losses.append(metrics.loss)
    mx.eval(base.loss, *losses)  # Test sync: materialize losses for Python assertions.

    for loss in losses:
        assert abs(float(loss.item()) - float(base.loss.item())) < 1e-6


def test_gspo_sequence_ratio_uses_length_normalized_log_likelihood() -> None:
    policy = mx.log(mx.array([[0.20, 0.50], [0.40, 0.25]], dtype=mx.float32))
    old_policy = mx.log(mx.array([[0.10, 0.25], [0.20, 0.50]], dtype=mx.float32))
    rewards = mx.array([1.0, 3.0], dtype=mx.float32)
    advantages = group_normalize_rewards(rewards, group_size=2)
    algorithm = GSPOAlgorithm(importance="sequence", clip_low=None, clip_high=None)

    metrics = algorithm.loss(
        policy_logprobs=policy,
        old_policy_logprobs=old_policy,
        reference_logprobs=policy,
        advantages=advantages,
        mask=mx.ones_like(policy),
        beta=0.0,
    )
    mx.eval(metrics.loss, metrics.mean_ratio)  # Test sync: materialize GSPO scalars.

    assert abs(float(metrics.mean_ratio.item()) - 1.5) < 1e-6
    assert abs(float(metrics.loss.item()) - 0.5) < 1e-6


def test_approximate_kl_is_zero_when_policies_match() -> None:
    logprobs = mx.log(mx.array([[0.3, 0.7]], dtype=mx.float32))

    kl = approximate_kl(logprobs, logprobs)
    mx.eval(kl)  # Test sync: materialize values before Python list assertion.

    assert kl.tolist() == [[0.0, 0.0]]
