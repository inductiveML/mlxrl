"""Experimental score centering, arXiv:2609.20807, equations 8--10.

This implementation uses group-centered rewards and token-mean reduction.
It is a single-turn REINFORCE variant, with an optional independent reference
KL penalty. It does not implement Tinker's proprietary StableReinforce loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxrl.algo.grpo import _token_loss_metrics, group_center_rewards
from mlxrl.algorithm import AlgorithmLossMetrics
from mlxrl.sampling import SamplerBatch


def score_centering_correction(
    trainer_head_logprobs: mx.array,
    sampler_head_logprobs: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Head-only scalar whose derivative estimates the sampler's mean score.

    Head probabilities retain their original mass. The trainer models the tail
    after rescaling it to the sampler's residual mass. Detach the entire residual
    coefficient, including its trainer-dependent tail scale (paper equation 8).
    """

    if trainer_head_logprobs.shape != sampler_head_logprobs.shape:
        raise ValueError("Sampler and trainer head shapes must match.")
    if not 0 < eps < 1:
        raise ValueError("eps must lie in (0, 1).")
    p = mx.exp(trainer_head_logprobs.astype(mx.float32))
    q = mx.exp(mx.stop_gradient(sampler_head_logprobs.astype(mx.float32)))
    p_tail = mx.maximum(1.0 - mx.sum(p, axis=-1, keepdims=True), eps)
    # An exactly exhausted sampler head has no tail; do not invent extra mass.
    q_tail = mx.maximum(1.0 - mx.sum(q, axis=-1, keepdims=True), 0.0)
    residual = mx.stop_gradient(q - (q_tail / p_tail) * p)
    return mx.sum(residual * trainer_head_logprobs, axis=-1)


@dataclass(frozen=True)
class ScoreCenteringAlgorithm:
    """Opt-in, critic-free score centering with captured sampler probabilities."""

    sampler_top_k: int = 128
    name: str = "score-centering"

    def __post_init__(self) -> None:
        if type(self.sampler_top_k) is not int or self.sampler_top_k < 1:
            raise ValueError("sampler_top_k must be a positive integer.")

    @property
    def token_mean_reduction(self) -> bool:
        return True

    def compute_advantages(self, rewards: mx.array, group_structure: int) -> mx.array:
        return group_center_rewards(rewards, group_structure)

    def filter_batch(self, batch: Any, group_structure: int) -> Any:
        del group_structure
        return batch

    def compute_loss(
        self,
        policy_logprobs: mx.array,
        old_policy_logprobs: mx.array,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        raise ValueError("Score centering requires sampler records and candidate trainer logprobs.")

    def compute_sampler_loss(
        self,
        policy_logprobs: mx.array,
        trainer_head_logprobs: mx.array,
        sampler: SamplerBatch,
        reference_logprobs: mx.array,
        advantages: mx.array,
        completion_mask: mx.array,
        beta: float,
    ) -> AlgorithmLossMetrics:
        correction = score_centering_correction(trainer_head_logprobs, sampler.topk_logprobs)
        token_loss = -advantages[:, None] * (policy_logprobs - correction)
        return _token_loss_metrics(
            token_policy_gradient=token_loss,
            ratio=mx.exp(policy_logprobs - sampler.logprobs),
            policy_logprobs=policy_logprobs,
            reference_logprobs=reference_logprobs,
            mask=completion_mask,
            beta=beta,
        )
