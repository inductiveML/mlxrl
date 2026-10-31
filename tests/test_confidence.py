from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs

from mlxrl.algo.grpo import DAPOAlgorithm, GRPOAlgorithm
from mlxrl.algo.score_centering import ScoreCenteringAlgorithm
from mlxrl.confidence import ConfidenceWarmStart, centered_confidence
from mlxrl.config import ConfidenceConfig
from mlxrl.data.confidence import ConfidenceStep, TokenSpan
from mlxrl.policy.logprobs import completion_logprobs_from_inputs, prepare_completion_logprob_inputs
from mlxrl.policy.model import LoRAConfig, inject_lora_adapters
from mlxrl.rollout.naive import Completion
from mlxrl.sampling import SamplerBatch
from mlxrl.train.grpo import GRPOBatch, batch_from_rollouts, grpo_metrics_from_batch, optimizer_step

pytestmark = pytest.mark.metal


def steps():
    return (
        ConfidenceStep(TokenSpan(0, 3), (TokenSpan(2, 3),)),
        ConfidenceStep(TokenSpan(3, 5), (TokenSpan(4, 5),)),
    )


def batch():
    lp = mx.log(mx.array([[0.1, 0.1, 0.2, 0.1, 0.8, 0.1]] * 4))
    return GRPOBatch(
        ((1,),) * 4,
        ((1, 2, 3, 4, 5, 6),) * 4,
        mx.array([0, 1, 1, 0]),
        mx.array([-1, 1, 1, -1]),
        lp,
        lp,
        mx.ones((4, 6)),
        confidence_logprobs=lp,
    )


def test_weighted_centering_uses_joint_result_probability_and_exclusion():
    units = (
        ConfidenceStep(TokenSpan(0, 4), (TokenSpan(2, 4),)),
        ConfidenceStep(TokenSpan(4, 6), (TokenSpan(5, 6),)),
        ConfidenceStep(TokenSpan(6, 8)),
    )
    # First result has probability .5*.4=.2; second .8. Two vs one credited tokens.
    cr, eligible, scored = centered_confidence(
        np.log([0.1, 0.1, 0.5, 0.4, 0.1, 0.8, 0.1, 0.1]).tolist(), units
    )
    np.testing.assert_allclose(cr, [-0.2, -0.2, 0, 0, 0.4, 0, 0, 0], atol=1e-12)
    assert sum(cr) == pytest.approx(0, abs=1e-12)
    assert eligible == [True, True, False, False, True, False, True, True]
    assert scored == 2


@pytest.mark.parametrize("units", [(), steps()[:1]])
def test_no_usable_relative_confidence_has_zero_credit(units):
    cr, _, _ = centered_confidence([math.log(0.2)] * 6, units)
    assert cr == [0] * 6


@pytest.mark.parametrize(
    "units",
    [
        (ConfidenceStep(TokenSpan(-1, 2)),),
        (ConfidenceStep(TokenSpan(0, 7)),),
        (ConfidenceStep(TokenSpan(0, 3), (TokenSpan(3, 4),)),),
        (ConfidenceStep(TokenSpan(0, 3)), ConfidenceStep(TokenSpan(2, 4))),
    ],
)
def test_bad_spans_are_rejected(units):
    with pytest.raises(ValueError, match="spans"):
        centered_confidence([-1.0] * 6, units)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
def test_invalid_confidence_logprobs_are_rejected(value):
    with pytest.raises(ValueError, match="log-probabilities"):
        centered_confidence([-1.0, -1.0, value, -1.0, -1.0, -1.0], steps())


def test_calibration_matches_hand_computation_and_leaves_value_answer_tokens_alone():
    controller = ConfidenceWarmStart(ConfidenceConfig(enabled=True))
    source = batch()
    changed, metrics = controller.apply(source, [steps()] * 4, update=1, group_size=2)
    # mean absolute unit credit = (.2+.2+.4)/3, mean |base advantage|=1.
    assert metrics.scale == pytest.approx(0.875 / (0.8 / 3), rel=2e-6)
    assert metrics.share == pytest.approx(0.875)
    assert metrics.mixed_groups == 2
    credit = np.asarray(changed.token_advantages) - np.asarray(source.advantages)[:, None]
    np.testing.assert_allclose(credit[:, [2, 4, 5]], 0, atol=0)
    np.testing.assert_allclose(credit.sum(axis=1), 0, atol=2e-7)
    assert source.token_advantages is None


def test_hold_scale_uniform_groups_resume_and_update_nine_exact_handoff():
    config = ConfidenceConfig(enabled=True)
    controller = ConfidenceWarmStart(config)
    source = batch()
    controller.apply(source, [steps()] * 4, update=1, group_size=2)
    uniform = replace(source, rewards=mx.zeros(4), advantages=mx.zeros(4))
    carried, metrics = controller.apply(uniform, [steps()] * 4, update=2, group_size=2)
    assert metrics.held and carried.token_advantages is not None
    assert bool(mx.any(carried.token_advantages != 0))
    for update in range(3, 9):
        controller.apply(source, [steps()] * 4, update=update, group_size=2)
    resumed = ConfidenceWarmStart(config)
    resumed.load_state_dict(controller.state_dict())
    missing_records = replace(source, confidence_logprobs=None)
    result, metrics = resumed.apply(missing_records, (), update=9, group_size=2)
    assert result is missing_records
    assert not metrics.active and metrics.scale == 0
    with pytest.raises(ValueError, match="consecutive"):
        resumed.apply(source, (), update=1, group_size=2)


def test_cap_applies_to_held_scale_and_no_credit_is_exact_noop():
    source = batch()
    config = ConfidenceConfig(enabled=True, p99_cap=0.25)
    controller = ConfidenceWarmStart(config)
    _, metrics = controller.apply(source, [steps()] * 4, update=1, group_size=2)
    assert metrics.capped and metrics.p99 == pytest.approx(0.25)
    controller.previous_scale = 1000
    _, metrics = controller.apply(source, [steps()] * 4, update=2, group_size=2)
    assert metrics.held and metrics.capped and metrics.p99 == pytest.approx(0.25)
    result, _ = controller.apply(source, [()] * 4, update=3, group_size=2)
    assert result is source
    result, _ = ConfidenceWarmStart().apply(source, (), update=1, group_size=2)
    assert result is source


def test_controller_rejects_missing_records_incomplete_groups_and_double_credit():
    controller = ConfidenceWarmStart(ConfidenceConfig(enabled=True))
    with pytest.raises(ValueError, match="record_confidence"):
        controller.apply(replace(batch(), confidence_logprobs=None), (), update=1, group_size=2)
    with pytest.raises(ValueError, match="complete prompt groups"):
        controller.apply(batch(), [steps()] * 4, update=1, group_size=3)
    result, _ = controller.apply(batch(), [steps()] * 4, update=1, group_size=2)
    with pytest.raises(ValueError, match="already"):
        controller.apply(result, [steps()] * 4, update=2, group_size=2)


@pytest.mark.parametrize("reduction", ["sequence_mean", "token_mean"])
def test_token_advantage_gradient_matches_ramp_formula(reduction):
    logp = mx.array([[-1.0, -2.0, -3.0], [-2.0, -1.0, 0.0]])
    old = mx.stop_gradient(logp)
    mask = mx.array([[1, 1, 1], [1, 1, 0]])
    advantages = mx.array([[0.2, -0.4, 0.3], [-0.2, 0.6, 0]])
    algorithm = GRPOAlgorithm(loss_reduction=reduction)

    def loss(lp):
        return algorithm.compute_loss(lp, old, lp, advantages, mask, 0).loss

    divisor = np.array([[6], [4]]) if reduction == "sequence_mean" else 5
    np.testing.assert_allclose(
        mx.grad(loss)(logp), -np.asarray(advantages) * mask / divisor, atol=1e-7
    )
    np.testing.assert_allclose(
        mx.grad(lambda a: algorithm.compute_loss(logp, old, logp, a, mask, 0).loss)(advantages),
        0,
        atol=0,
    )


def model(quantized=False):
    mx.random.seed(91)
    m = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=16,
            max_position_embeddings=128,
        )
    )
    if quantized:
        nn.quantize(m, group_size=32, bits=4)
    inject_lora_adapters(m, LoRAConfig(rank=4, scale=2, dropout=0))
    mx.eval(m.parameters())
    m.eval()
    return m


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_raw_confidence_matches_fp32_full_forward_and_preserves_old_logprobs(quantized, fallback):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model(quantized)

        def __call__(self, x):
            return self.inner(x)

    m = Wrapper() if fallback else model(quantized)
    inputs = prepare_completion_logprob_inputs(((1, 2), (3,)), ((4, 5), (6,)), 0)
    plain = completion_logprobs_from_inputs(m, inputs)
    recorded = completion_logprobs_from_inputs(m, inputs, record_confidence=True)
    np.testing.assert_array_equal(recorded.logprobs, plain.logprobs)
    logits = m(inputs.input_ids).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    expected = mx.array([[logp[0, 1, 4].item(), logp[0, 2, 5].item()], [logp[1, 0, 6].item(), 0.0]])
    assert recorded.confidence_logprobs is not None
    np.testing.assert_allclose(recorded.confidence_logprobs, expected, atol=2e-6)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("reduction", ["token_mean", "sequence_mean"])
def test_conf_whole_batch_and_microbatch_updates_match(quantized, centered, reduction):
    whole, micro = model(quantized), model(quantized)
    algorithm = (
        ScoreCenteringAlgorithm(sampler_top_k=2, normalize_rewards=True, loss_reduction=reduction)
        if centered
        else GRPOAlgorithm(loss_reduction=reduction)
    )
    # Prepare with GRPO to supply fixed synthetic sampler records below.
    completions = [
        Completion(i // 2, i % 2, (1, 2), tuple(range(1, n + 1)), (), "")
        for i, n in enumerate([6, 5, 7, 5])
    ]
    source = batch_from_rollouts(
        whole,
        completions,
        [0, 1, 1, 0],
        2,
        0,
        GRPOAlgorithm(),
        compute_reference=False,
        record_confidence=True,
    )
    if centered:
        shape = (*source.mask.shape, 2)
        source = replace(
            source,
            sampler=SamplerBatch(
                source.old_policy_logprobs,
                mx.zeros(shape, dtype=mx.int32).at[:, :, 1].add(1),
                mx.full(shape, math.log(0.1)),
            ),
        )
    controller = ConfidenceWarmStart(ConfidenceConfig(enabled=True))
    prepared, _ = controller.apply(source, [steps()] * 4, update=1, group_size=2)
    assert prepared.token_advantages is not None
    a = optimizer_step(whole, optim.SGD(0.01), prepared, 0, 0, algorithm)
    b = optimizer_step(micro, optim.SGD(0.01), prepared, 0, 0, algorithm, micro_batch_size=1)
    assert math.isfinite(a.loss)
    assert a.loss == pytest.approx(b.loss, abs=2e-6)
    for key in ("policy_gradient_loss", "kl", "mean_ratio", "clip_fraction"):
        assert getattr(a, key) == pytest.approx(getattr(b, key), rel=2e-6, abs=2e-6)
    for (_, x), (_, y) in zip(
        tree_flatten(whole.parameters()), tree_flatten(micro.parameters()), strict=True
    ):
        np.testing.assert_allclose(cast(Any, x), cast(Any, y), atol=2e-6)
    with pytest.raises(ValueError, match="does not support"):
        grpo_metrics_from_batch(whole, prepared, 0, 0, DAPOAlgorithm())
