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
from mlx_lm.sample_utils import make_sampler

from mlxrl.algo.score_centering import ScoreCenteringAlgorithm, score_centering_correction
from mlxrl.policy.logprobs import completion_logprobs_from_inputs, prepare_completion_logprob_inputs
from mlxrl.policy.model import LoRAConfig, inject_lora_adapters
from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
from mlxrl.rollout.optimized import generate_prefix_cached_group_rollouts
from mlxrl.sampling import (
    SamplerToken,
    capture_sampler_tokens,
    sampler_batch_from_traces,
    sampler_logprobs,
)
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

pytestmark = pytest.mark.metal


def test_exact_centered_gradient_matches_enumerated_covariance() -> None:
    q = mx.array([0.1, 0.2, 0.3, 0.4])
    advantages = mx.array([1.0, -1.0, 2.0, -2.0])
    logits = mx.array([0.2, -0.1, 0.7, 1.1])

    def loss(theta: mx.array) -> mx.array:
        logp = theta - mx.logsumexp(theta)
        correction = score_centering_correction(logp, mx.log(q))
        return -mx.sum(q * advantages * (logp - correction))

    actual = mx.grad(loss)(logits)
    expected = -q * (advantages - mx.sum(q * advantages))
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_constant_reward_has_no_expected_drift_under_mismatch() -> None:
    q = mx.array([0.6, 0.1, 0.2, 0.1])
    logits = mx.array([-0.4, 0.2, 0.1, 1.0])

    def loss(theta: mx.array, centered: bool) -> mx.array:
        logp = theta - mx.logsumexp(theta)
        correction = score_centering_correction(logp, mx.log(q)) if centered else 0.0
        return -mx.sum(q * (logp - correction))

    plain = mx.grad(lambda theta: loss(theta, False))(logits)
    centered = mx.grad(lambda theta: loss(theta, True))(logits)
    assert float(mx.max(mx.abs(plain))) > 0.1
    np.testing.assert_allclose(centered, mx.zeros_like(centered), atol=2e-6)


def test_matched_sampler_correction_is_zero_in_value_and_gradient() -> None:
    logits = mx.array([0.3, -0.2, 1.1, 0.1])
    fixed_q = mx.stop_gradient(mx.softmax(logits))
    ids = mx.array([0, 2])

    def correction(theta: mx.array) -> mx.array:
        logp = theta - mx.logsumexp(theta)
        return score_centering_correction(logp[ids], mx.log(fixed_q[ids]))

    assert abs(float(correction(logits))) < 1e-6
    np.testing.assert_allclose(mx.grad(correction)(logits), mx.zeros_like(logits), atol=1e-6)


def test_topk_tail_gradient_matches_full_reconstructed_sampler() -> None:
    logits = mx.array([0.3, -0.7, 0.1, 0.8, -0.1])
    ids = mx.array([0, 3])
    q_head = mx.array([0.2, 0.4])

    def full(theta: mx.array) -> mx.array:
        logp = theta - mx.logsumexp(theta)
        p = mx.exp(logp)
        rho = (1 - mx.sum(q_head)) / (1 - mx.sum(p[ids]))
        q = rho * p
        q = q.at[ids].add(q_head - q[ids])
        return mx.sum(mx.stop_gradient(q) * logp)

    def reduced(theta: mx.array) -> mx.array:
        logp = theta - mx.logsumexp(theta)
        return score_centering_correction(logp[ids], mx.log(q_head))

    np.testing.assert_allclose(mx.grad(reduced)(logits), mx.grad(full)(logits), atol=2e-6)


@pytest.mark.parametrize("logits", [[80.0, -80.0, 0.0], [0.0, 0.0, 0.0]])
def test_near_zero_tail_and_zero_probability_candidates_are_finite(logits) -> None:
    theta = mx.array(logits)
    q_logp = mx.array([0.0, -float("inf"), -float("inf")])

    def loss(x: mx.array) -> mx.array:
        logp = x - mx.logsumexp(x)
        return -(logp[0] - score_centering_correction(logp, q_logp))

    assert math.isfinite(float(loss(theta)))
    assert bool(mx.all(mx.isfinite(mx.grad(loss)(theta))))


@pytest.mark.parametrize(
    "temperature,top_p,min_p,top_k,kept",
    [
        (1.0, 1.0, 0.0, 0, 5),
        (0.7, 1.0, 0.0, 0, 5),
        (0.5, 0.8, 0.0, 0, 3),
        (0.7, 0.95, 0.4, 2, 2),
        (0.0, 0.8, 0.4, 2, 1),
    ],
)
def test_capture_matches_actual_filtered_categorical_law(
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    kept: int,
) -> None:
    raw = mx.log(mx.array([[0.4, 0.3, 0.15, 0.1, 0.05]]))
    actual = sampler_logprobs(raw, temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k)
    # Hand-selected supports and NumPy arithmetic are independent of MLX filters.
    if temperature == 0:
        expected = mx.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    else:
        probabilities = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        expected = probabilities ** (1 / temperature)
        expected[kept:] = 0
        expected = (expected / expected.sum())[None, :]
    np.testing.assert_allclose(mx.exp(actual), expected, atol=1e-6)
    assert abs(float(mx.sum(mx.exp(actual))) - 1) < 1e-6
    record = capture_sampler_tokens(
        raw,
        mx.array([0]),
        record_top_k=3,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )[0]
    assert abs(record.logprob - float(actual[0, 0])) < 1e-6
    for token, lp in zip(record.topk_ids, record.topk_logprobs, strict=True):
        assert math.exp(lp) == pytest.approx(float(expected[0, token]), abs=1e-6)
    if temperature == 0.7:
        assert abs(record.logprob - float(raw[0, 0])) > 0.05
    mx.random.seed(2333)
    draws = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)(
        mx.broadcast_to(raw, (20000, 5))
    )
    frequencies = np.bincount(np.array(draws), minlength=5) / 20000
    np.testing.assert_allclose(frequencies, np.array(expected)[0], atol=0.015)


class TinyTokenizer:
    chat_template = None
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3] if text == "a" else [4, 5]

    def decode(self, tokens) -> str:
        return " ".join(str(t) for t in tokens)


def test_selected_token_outside_recorded_head_retains_its_actual_probability() -> None:
    raw = mx.log(mx.array([[0.4, 0.3, 0.15, 0.1, 0.05]]))
    record = capture_sampler_tokens(
        raw,
        mx.array([4]),
        record_top_k=2,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )[0]
    assert 4 not in record.topk_ids
    assert math.exp(record.logprob) == pytest.approx(0.05, abs=1e-6)
    assert sum(math.exp(p) for p in record.topk_logprobs) == pytest.approx(0.7, abs=1e-6)
    batch = sampler_batch_from_traces([[record]], [[4]])
    assert float(batch.logprobs[0, 0]) == pytest.approx(math.log(0.05), abs=1e-6)


def _model(seed: int = 11, quantized: bool = False) -> Model:
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64 if quantized else 16,
            num_hidden_layers=1,
            intermediate_size=128 if quantized else 32,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=16,
            max_position_embeddings=128,
        )
    )
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        inject_lora_adapters(model, LoRAConfig(rank=4, scale=2, dropout=0))
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.mark.parametrize("engine", ["naive", "prefix", "batched", "compiled"])
@pytest.mark.parametrize("early_eos", [False, True])
def test_capture_preserves_seeded_tokens_and_eos_alignment(engine: str, early_eos: bool) -> None:
    model = _model()
    tokenizer: Any = TinyTokenizer()
    if early_eos:
        tokenizer.eos_token_ids = list(range(16))
    kwargs: dict[str, Any] = dict(
        model=model, tokenizer=tokenizer, prompts=["a", "b"], group_size=2, seed=9
    )
    sampling = SamplingConfig(max_tokens=3, temperature=0.7, top_p=0.9, top_k=8)
    fn = generate_group_rollouts if engine == "naive" else generate_prefix_cached_group_rollouts
    if engine != "naive":
        kwargs.update(
            batch_groups=engine in {"batched", "compiled"}, compile_decode_step=engine == "compiled"
        )
    before = fn(**kwargs, config=sampling)
    after = fn(**kwargs, config=replace(sampling, record_sampler_top_k=4))
    assert [r.completion_tokens for r in before] == [r.completion_tokens for r in after]
    for plain, captured in zip(before, after, strict=True):
        assert plain.sampler_trace is None
        assert captured.sampler_trace is not None
        assert len(captured.sampler_trace) == len(captured.completion_tokens)
        assert len(captured.completion_tokens) == (1 if early_eos else 3)
        np.testing.assert_allclose(plain.old_policy_logprobs, captured.old_policy_logprobs)
        assert all(math.isfinite(t.logprob) for t in captured.sampler_trace)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_candidate_logprobs_and_gradients_match_direct_full_forward(
    fallback: bool,
    quantized: bool,
) -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = _model(quantized=quantized)

        def __call__(self, tokens: mx.array) -> mx.array:
            return self.inner(tokens)

    model = Wrapper() if fallback else _model(quantized=quantized)
    prompts, completions = ((1, 2), (3,)), ((4, 5), (6,))
    inputs = prepare_completion_logprob_inputs(prompts, completions)
    ids = mx.array([[[1, 7, 11], [0, 4, 8]], [[3, 9, 15], [0, 0, 0]]])
    actual = completion_logprobs_from_inputs(model, inputs, candidate_token_ids=ids)
    assert actual.candidate_logprobs is not None
    full = model(inputs.input_ids).astype(mx.float32)
    full = full - mx.logsumexp(full, axis=-1, keepdims=True)
    expected = mx.stack([full[0, 1], full[0, 2], full[1, 0], full[1, 0]]).reshape(2, 2, 16)
    expected = mx.take_along_axis(expected, ids, axis=-1)
    np.testing.assert_allclose(actual.candidate_logprobs, expected, atol=2e-6)

    def gathered_loss(m: nn.Module) -> mx.array:
        values = completion_logprobs_from_inputs(m, inputs, candidate_token_ids=ids)
        assert values.candidate_logprobs is not None
        return mx.sum(values.candidate_logprobs * inputs.mask[..., None])

    def explicit_loss(m: nn.Module) -> mx.array:
        logits = m(inputs.input_ids).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        positions = mx.stack([logp[0, 1], logp[0, 2], logp[1, 0], logp[1, 0]])
        positions = positions.reshape(2, 2, 16)
        return mx.sum(mx.take_along_axis(positions, ids, axis=-1) * inputs.mask[..., None])

    _, gathered_grad = nn.value_and_grad(model, gathered_loss)(model)
    _, explicit_grad = nn.value_and_grad(model, explicit_loss)(model)
    for (name, actual_grad), (other_name, expected_grad) in zip(
        tree_flatten(gathered_grad), tree_flatten(explicit_grad), strict=True
    ):
        assert name == other_name
        np.testing.assert_allclose(cast(Any, actual_grad), cast(Any, expected_grad), atol=2e-5)


def test_score_centering_optimizer_microbatch_matches_whole_batch_with_padding() -> None:
    whole, micro = _model(), _model()
    algorithm = ScoreCenteringAlgorithm(sampler_top_k=4)
    completions = list(
        generate_group_rollouts(
            whole,
            TinyTokenizer(),
            ["a", "b"],
            2,
            SamplingConfig(max_tokens=3, record_sampler_top_k=4),
            seed=7,
        )
    )
    assert completions[1].sampler_trace is not None
    completions[1] = replace(
        completions[1],
        completion_tokens=completions[1].completion_tokens[:1],
        sampler_trace=completions[1].sampler_trace[:1],
    )
    batch = batch_from_rollouts(
        whole, completions, [1, -1, -1, 1], 2, 0, algorithm, compute_reference=False
    )
    a = optimizer_step(whole, optim.SGD(0.01), batch, 0, 0, algorithm)
    b = optimizer_step(micro, optim.SGD(0.01), batch, 0, 0, algorithm, micro_batch_size=1)
    assert math.isfinite(a.loss)
    assert a.loss == pytest.approx(b.loss, abs=2e-6)
    for (_, x), (_, y) in zip(
        tree_flatten(whole.parameters()), tree_flatten(micro.parameters()), strict=True
    ):
        np.testing.assert_allclose(cast(Any, x), cast(Any, y), atol=2e-6)


def test_missing_and_malformed_sampler_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="captured sampler"):
        sampler_batch_from_traces([None], [[1]])
    with pytest.raises(ValueError, match="one sampler record"):
        sampler_batch_from_traces([[SamplerToken(0, (1,), (0,))]], [[1, 2]])
    with pytest.raises(ValueError, match="distinct"):
        sampler_batch_from_traces([[SamplerToken(0, (1, 1), (0, -math.inf))]], [[1]])
    with pytest.raises(ValueError, match="disagree"):
        sampler_batch_from_traces([[SamplerToken(-1, (1,), (-2,))]], [[1]])
