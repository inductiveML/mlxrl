from __future__ import annotations

import math

import mlx.core as mx

from mlxrl.algo.grpo import (
    DAPOAlgorithm,
    DrGRPOAlgorithm,
    GSPOAlgorithm,
    RLOOAlgorithm,
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


def test_prompt_positions_are_zero_contribution_to_loss_and_grad() -> None:
    policy = mx.log(mx.array([[0.90, 0.10, 0.30, 0.70]], dtype=mx.float32))
    old_policy = mx.log(mx.array([[0.80, 0.20, 0.25, 0.50]], dtype=mx.float32))
    reference = mx.log(mx.array([[0.05, 0.95, 0.40, 0.60]], dtype=mx.float32))
    advantages = mx.array([1.25], dtype=mx.float32)
    completion_mask = mx.array([[0.0, 0.0, 1.0, 1.0]], dtype=mx.float32)
    beta = 0.3

    masked = grpo_loss(
        policy_logprobs=policy,
        old_policy_logprobs=old_policy,
        reference_logprobs=reference,
        advantages=advantages,
        mask=completion_mask,
        beta=beta,
    )
    completion_only = grpo_loss(
        policy_logprobs=policy[:, 2:],
        old_policy_logprobs=old_policy[:, 2:],
        reference_logprobs=reference[:, 2:],
        advantages=advantages,
        mask=mx.ones((1, 2), dtype=mx.float32),
        beta=beta,
    )

    def loss_fn(policy_logprobs: mx.array) -> mx.array:
        return grpo_loss(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=old_policy,
            reference_logprobs=reference,
            advantages=advantages,
            mask=completion_mask,
            beta=beta,
        ).loss

    policy_grad = mx.grad(loss_fn)(policy)
    mx.eval(  # Test sync: materialize masked loss and gradient for exact assertions.
        masked.loss,
        completion_only.loss,
        policy_grad,
    )

    assert float(masked.loss.item()) == float(completion_only.loss.item())
    assert policy_grad[:, :2].tolist() == [[0.0, 0.0]]
    assert any(abs(value) > 0.0 for value in policy_grad[:, 2:].tolist()[0])


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


def test_dapo_loss_and_gradient_match_hand_computed_clipped_example() -> None:
    old_policy = mx.zeros((2, 2), dtype=mx.float32)
    ratios = mx.array([[1.1, 1.5], [0.95, 0.5]], dtype=mx.float32)
    policy = mx.log(ratios)
    advantages = mx.array([1.0, -1.0], dtype=mx.float32)
    algorithm = DAPOAlgorithm(clip_low=0.2, clip_high=0.2)

    def loss_fn(policy_logprobs: mx.array) -> mx.array:
        return algorithm.compute_loss(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=old_policy,
            reference_logprobs=policy_logprobs,
            advantages=advantages,
            completion_mask=mx.ones_like(policy_logprobs),
            beta=0.0,
        ).loss

    loss = loss_fn(policy)
    gradient = mx.grad(loss_fn)(policy)
    expected_loss = (-1.1 - 1.2 + 0.95 + 0.8) / 4.0
    expected_gradient = mx.array([[-1.1, 0.0], [0.95, 0.0]], dtype=mx.float32) / 4.0
    max_gradient_error = mx.max(mx.abs(gradient - expected_gradient))
    mx.eval(  # Test sync: materialize DAPO toy loss and gradient.
        loss,
        max_gradient_error,
    )

    assert abs(float(loss.item()) - expected_loss) < 1e-6
    assert float(max_gradient_error.item()) < 1e-6


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


def test_gspo_sequence_loss_and_gradient_match_hand_computed_example() -> None:
    old_policy = mx.zeros((2, 2), dtype=mx.float32)
    policy = mx.log(mx.array([[2.0, 2.0], [2.0, 0.5]], dtype=mx.float32))
    advantages = mx.array([-1.0, 1.0], dtype=mx.float32)
    algorithm = GSPOAlgorithm(importance="sequence", clip_low=None, clip_high=None)

    def loss_fn(policy_logprobs: mx.array) -> mx.array:
        return algorithm.compute_loss(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=old_policy,
            reference_logprobs=policy_logprobs,
            advantages=advantages,
            completion_mask=mx.ones_like(policy_logprobs),
            beta=0.0,
        ).loss

    loss = loss_fn(policy)
    gradient = mx.grad(loss_fn)(policy)
    expected_gradient = mx.array([[0.5, 0.5], [-0.25, -0.25]], dtype=mx.float32)
    max_gradient_error = mx.max(mx.abs(gradient - expected_gradient))
    mx.eval(  # Test sync: materialize GSPO toy loss and sequence-ratio gradient.
        loss,
        max_gradient_error,
    )

    assert abs(float(loss.item()) - 0.5) < 1e-6
    assert float(max_gradient_error.item()) < 1e-6


def test_rloo_advantage_loss_and_gradient_match_hand_computed_example() -> None:
    rewards = mx.array([1.0, 2.0, 4.0], dtype=mx.float32)
    policy = mx.log(
        mx.array([[0.50, 0.25], [0.20, 0.10], [0.40, 0.20]], dtype=mx.float32)
    )
    algorithm = RLOOAlgorithm()
    advantages = algorithm.compute_advantages(rewards, group_structure=3)
    expected_advantages = mx.array([-2.0, -0.5, 2.5], dtype=mx.float32)

    def loss_fn(policy_logprobs: mx.array) -> mx.array:
        return algorithm.compute_loss(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=policy_logprobs,
            reference_logprobs=policy_logprobs,
            advantages=advantages,
            completion_mask=mx.ones_like(policy_logprobs),
            beta=0.0,
        ).loss

    loss = loss_fn(policy)
    gradient = mx.grad(loss_fn)(policy)
    expected_loss = -(
        math.log(0.50) * -2.0
        + math.log(0.25) * -2.0
        + math.log(0.20) * -0.5
        + math.log(0.10) * -0.5
        + math.log(0.40) * 2.5
        + math.log(0.20) * 2.5
    ) / 6.0
    expected_gradient = -expected_advantages[:, None] / 6.0
    max_advantage_error = mx.max(mx.abs(advantages - expected_advantages))
    max_gradient_error = mx.max(mx.abs(gradient - expected_gradient))
    mx.eval(  # Test sync: materialize RLOO toy advantage/loss/gradient.
        loss,
        max_advantage_error,
        max_gradient_error,
    )

    assert float(max_advantage_error.item()) == 0.0
    assert abs(float(loss.item()) - expected_loss) < 1e-6
    assert float(max_gradient_error.item()) < 1e-6


def test_approximate_kl_is_zero_when_policies_match() -> None:
    logprobs = mx.log(mx.array([[0.3, 0.7]], dtype=mx.float32))

    kl = approximate_kl(logprobs, logprobs)
    mx.eval(kl)  # Test sync: materialize values before Python list assertion.

    assert kl.tolist() == [[0.0, 0.0]]


def test_beta_zero_loss_and_grad_ignore_reference_logprobs() -> None:
    policy = mx.log(mx.array([[0.20, 0.50], [0.40, 0.25]], dtype=mx.float32))
    old_policy = mx.log(mx.array([[0.18, 0.52], [0.43, 0.20]], dtype=mx.float32))
    reference_a = mx.log(mx.array([[0.25, 0.40], [0.50, 0.20]], dtype=mx.float32))
    reference_b = mx.zeros_like(reference_a)
    advantages = mx.array([-1.0, 1.0], dtype=mx.float32)
    mask = mx.ones_like(policy)

    def loss_with_reference(reference: mx.array):
        def loss_fn(policy_logprobs: mx.array) -> mx.array:
            return grpo_loss(
                policy_logprobs=policy_logprobs,
                old_policy_logprobs=old_policy,
                reference_logprobs=reference,
                advantages=advantages,
                mask=mask,
                beta=0.0,
            ).loss

        return loss_fn(policy), mx.grad(loss_fn)(policy)

    loss_a, grad_a = loss_with_reference(reference_a)
    loss_b, grad_b = loss_with_reference(reference_b)
    loss_error = mx.abs(loss_a - loss_b)
    grad_error = mx.max(mx.abs(grad_a - grad_b))
    mx.eval(  # Test sync: materialize beta-zero loss/grad comparison.
        loss_error,
        grad_error,
    )

    assert float(loss_error.item()) == 0.0
    assert float(grad_error.item()) == 0.0
