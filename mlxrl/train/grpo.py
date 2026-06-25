"""One-step GRPO training over LoRA adapter parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from mlxrl.algorithm import Algorithm, AlgorithmLossMetrics
from mlxrl.policy.logprobs import (
    CompletionLogprobInputs,
    CompletionLogprobs,
    completion_logprobs,
    completion_logprobs_from_inputs,
    dual_logprobs,
    prepare_completion_logprob_inputs,
)
from mlxrl.policy.model import enable_grad_checkpointing
from mlxrl.rollout.naive import Completion
from mlxrl.train.reference import reference_logprobs_for_loss


@dataclass(frozen=True)
class GRPOBatch:
    """Reference data for one GRPO optimizer step."""

    prompt_token_ids: tuple[tuple[int, ...], ...]
    completion_token_ids: tuple[tuple[int, ...], ...]
    rewards: mx.array
    advantages: mx.array
    old_policy_logprobs: mx.array
    reference_logprobs: mx.array
    mask: mx.array
    reference_is_policy: bool = False
    logprob_inputs: CompletionLogprobInputs | None = None


@dataclass(frozen=True)
class StepMetrics:
    """Python scalar diagnostics emitted after one optimizer step."""

    loss: float
    policy_gradient_loss: float
    kl: float
    mean_ratio: float
    clip_fraction: float
    mean_reward: float


def _slice_batch(batch: GRPOBatch, start: int, end: int) -> GRPOBatch:
    """Slice a GRPO batch and trim right-padding to the chunk completion width."""

    completion_token_ids = batch.completion_token_ids[start:end]
    if not completion_token_ids:
        raise ValueError("Cannot slice an empty micro-batch.")
    width = max(len(completion) for completion in completion_token_ids)
    logprob_inputs = (
        _slice_logprob_inputs(batch.logprob_inputs, batch, start, end, width)
        if batch.logprob_inputs is not None
        else None
    )
    return GRPOBatch(
        prompt_token_ids=batch.prompt_token_ids[start:end],
        completion_token_ids=completion_token_ids,
        rewards=batch.rewards[start:end],
        advantages=batch.advantages[start:end],
        old_policy_logprobs=batch.old_policy_logprobs[start:end, :width],
        reference_logprobs=batch.reference_logprobs[start:end, :width],
        mask=batch.mask[start:end, :width],
        reference_is_policy=batch.reference_is_policy,
        logprob_inputs=logprob_inputs,
    )


def _completion_token_count(batch: GRPOBatch) -> int:
    """Count valid completion tokens from host-side rollout metadata."""

    return sum(len(completion) for completion in batch.completion_token_ids)


def _slice_logprob_inputs(
    inputs: CompletionLogprobInputs,
    batch: GRPOBatch,
    start: int,
    end: int,
    completion_width: int,
) -> CompletionLogprobInputs:
    """Slice and trim prepared full-forward tensors for one micro-batch."""

    target_width = max(
        len(prompt) + len(completion) - 1
        for prompt, completion in zip(
            batch.prompt_token_ids[start:end],
            batch.completion_token_ids[start:end],
            strict=True,
        )
    )
    return CompletionLogprobInputs(
        input_ids=inputs.input_ids[start:end, :target_width],
        target_ids=inputs.target_ids[start:end, :target_width],
        gather_indices=inputs.gather_indices[start:end, :completion_width],
        mask=inputs.mask[start:end, :completion_width],
    )


def batch_from_rollouts(
    model: nn.Module,
    completions: Sequence[Completion],
    rewards: Sequence[float],
    group_size: int,
    pad_token_id: int,
    algorithm: Algorithm,
    use_checkpoint: bool = False,
    compute_reference: bool = True,
) -> GRPOBatch:
    """Compute old policy/ref logprobs and group-normalized advantages."""

    if len(completions) != len(rewards):
        raise ValueError("completions and rewards must have the same length.")
    if not completions:
        raise ValueError("At least one completion is required.")
    if use_checkpoint:
        enable_grad_checkpointing(model)
    prompt_token_ids = tuple(completion.prompt_tokens for completion in completions)
    completion_token_ids = tuple(completion.completion_tokens for completion in completions)
    logprob_inputs = prepare_completion_logprob_inputs(
        prompt_token_ids,
        completion_token_ids,
        pad_token_id=pad_token_id,
    )
    dual = dual_logprobs(
        model,
        prompt_token_ids,
        completion_token_ids,
        pad_token_id,
        use_checkpoint=use_checkpoint,
        compute_reference=compute_reference,
        prepared_inputs=logprob_inputs,
    )
    if compute_reference:
        mx.eval(  # Logprob sync: freeze old-policy/ref logprobs before adapter mutation.
            dual.policy,
            dual.reference,
            dual.mask,
        )
    else:
        mx.eval(  # Logprob sync: freeze old-policy logprobs before adapter mutation.
            dual.policy,
            dual.mask,
        )
    reward_array = mx.array(list(rewards), dtype=mx.float32)
    advantages = algorithm.compute_advantages(reward_array, group_structure=group_size)
    batch = GRPOBatch(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        rewards=reward_array,
        advantages=advantages,
        old_policy_logprobs=mx.stop_gradient(dual.policy),
        reference_logprobs=mx.stop_gradient(dual.reference),
        mask=dual.mask,
        reference_is_policy=not compute_reference,
        logprob_inputs=logprob_inputs,
    )
    return algorithm.filter_batch(batch, group_structure=group_size)


def old_policy_logprobs_from_rollouts(
    completions: Sequence[Completion],
) -> CompletionLogprobs:
    """Pad rollout-captured old-policy logprobs into the training tensor shape."""

    if not completions:
        raise ValueError("At least one completion is required.")
    max_completion_len = max(len(completion.completion_tokens) for completion in completions)
    if max_completion_len == 0:
        raise ValueError("At least one completion token is required.")

    logprob_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    for completion in completions:
        token_count = len(completion.completion_tokens)
        if len(completion.old_policy_logprobs) != token_count:
            raise ValueError(
                "Each completion must carry one old-policy logprob per token."
            )
        pad_count = max_completion_len - token_count
        logprob_rows.append(
            list(completion.old_policy_logprobs) + [0.0] * pad_count
        )
        mask_rows.append([1.0] * token_count + [0.0] * pad_count)

    return CompletionLogprobs(
        logprobs=mx.array(logprob_rows, dtype=mx.float32),
        mask=mx.array(mask_rows, dtype=mx.float32),
    )


def grpo_metrics_from_batch(
    model: nn.Module,
    batch: GRPOBatch,
    beta: float,
    pad_token_id: int,
    algorithm: Algorithm,
    use_checkpoint: bool = False,
) -> AlgorithmLossMetrics:
    """Recompute policy logprobs and evaluate GRPO metrics."""

    if use_checkpoint:
        enable_grad_checkpointing(model)
    current = (
        completion_logprobs_from_inputs(
            model,
            batch.logprob_inputs,
            use_checkpoint=use_checkpoint,
        )
        if batch.logprob_inputs is not None
        else completion_logprobs(
            model,
            batch.prompt_token_ids,
            batch.completion_token_ids,
            pad_token_id,
            use_checkpoint=use_checkpoint,
        )
    )
    reference_logprobs = reference_logprobs_for_loss(
        current.logprobs,
        batch.reference_logprobs,
        beta=beta,
        reference_is_policy=batch.reference_is_policy,
        batch_kind="GRPO batch",
    )
    return algorithm.compute_loss(
        policy_logprobs=current.logprobs,
        old_policy_logprobs=batch.old_policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=batch.advantages,
        completion_mask=batch.mask,
        beta=beta,
    )


def optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: GRPOBatch,
    beta: float,
    pad_token_id: int,
    algorithm: Algorithm,
    use_checkpoint: bool = False,
    micro_batch_size: int = 0,
) -> StepMetrics:
    """Run value_and_grad over currently trainable adapter parameters once."""

    if micro_batch_size < 0:
        raise ValueError("micro_batch_size must be non-negative.")
    num_completions = len(batch.completion_token_ids)
    chunked = 0 < micro_batch_size < num_completions
    if chunked and not algorithm.token_mean_reduction:
        raise ValueError(
            "micro_batch_size currently supports token-mean policy losses only; "
            f"{algorithm.name} uses sequence-level reduction."
        )
    model.train()

    def loss_fn(
        model: nn.Module,
        sub: GRPOBatch,
    ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array, mx.array]]:
        metrics = grpo_metrics_from_batch(
            model,
            sub,
            beta,
            pad_token_id,
            use_checkpoint=use_checkpoint,
            algorithm=algorithm,
        )
        return metrics.loss, (
            metrics.policy_gradient_loss,
            metrics.kl,
            metrics.mean_ratio,
            metrics.clip_fraction,
        )

    gradients: Any | None
    if not chunked:
        (loss, (policy_gradient_loss, kl, mean_ratio, clip_fraction)), gradients = (
            nn.value_and_grad(model, lambda m: loss_fn(m, batch))(model)
        )
    else:
        total_tokens = _completion_token_count(batch)
        if total_tokens <= 0:
            raise ValueError("Cannot micro-batch a GRPO batch with no valid tokens.")
        gradients = None
        loss = policy_gradient_loss = kl = mean_ratio = clip_fraction = mx.array(
            0.0,
            dtype=mx.float32,
        )
        for start in range(0, num_completions, micro_batch_size):
            end = min(start + micro_batch_size, num_completions)
            sub = _slice_batch(batch, start, end)
            weight = _completion_token_count(sub) / total_tokens
            (sub_loss, sub_aux), sub_grad = nn.value_and_grad(
                model,
                lambda m, _s=sub: loss_fn(m, _s),
            )(model)
            sub_grad = tree_map(
                lambda gradient, _weight=weight: gradient * _weight,
                sub_grad,
            )
            gradients = (
                sub_grad
                if gradients is None
                else tree_map(lambda left, right: left + right, gradients, sub_grad)
            )
            loss = loss + sub_loss * weight
            policy_gradient_loss = policy_gradient_loss + sub_aux[0] * weight
            kl = kl + sub_aux[1] * weight
            mean_ratio = mean_ratio + sub_aux[2] * weight
            clip_fraction = clip_fraction + sub_aux[3] * weight
            mx.eval(  # Micro-batch sync: free this chunk's backward graph before next chunk.
                gradients,
                loss,
                policy_gradient_loss,
                kl,
                mean_ratio,
                clip_fraction,
            )
    if gradients is None:
        raise RuntimeError("Micro-batch gradient accumulation produced no gradients.")
    mean_reward = mx.mean(batch.rewards)
    mx.eval(  # Optimizer pre-step sync: freeze gradients/diagnostics before weight mutation.
        gradients,
        loss,
        policy_gradient_loss,
        kl,
        mean_ratio,
        clip_fraction,
        mean_reward,
    )
    optimizer.update(model, gradients)
    mx.eval(  # Optimizer sync: materialize updated adapter weights and optimizer state.
        model.state,
        optimizer.state,
    )
    model.eval()
    return StepMetrics(
        loss=float(loss.item()),
        policy_gradient_loss=float(policy_gradient_loss.item()),
        kl=float(kl.item()),
        mean_ratio=float(mean_ratio.item()),
        clip_fraction=float(clip_fraction.item()),
        mean_reward=float(mean_reward.item()),
    )


def reward_trend(values: Sequence[float], window: int = 5) -> tuple[float, float]:
    """Return first-window and last-window means for a short sanity run."""

    if not values:
        raise ValueError("At least one reward value is required.")
    window = max(1, min(window, len(values)))
    first = sum(values[:window]) / window
    last = sum(values[-window:]) / window
    return first, last
