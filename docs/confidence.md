# Experimental CONF warm start

CONF is an optional token-credit modifier. The implementation is based on the
CONF-5/6 mechanism developed in the RAMP project, with independently written MLX
code. It uses no additional model, oracle step labels or trainable critic.

## Mathematics and invariants

For each scored reasoning segment, confidence is the mean probability of its
written result spans. A span's probability is `exp(sum(raw_logprobs))`, including
all tokens of a multi-token result. These are current-policy, temperature-1,
unfiltered probabilities, normalized in fp32 during batch preparation. They are
separate from score centering's actual sampler probabilities.

Within each completion, subtract the confidence mean weighted by eligible token
count. The extra advantage sums to zero over credited tokens. Result spans are
always excluded; additional exclusions and final-answer boundaries come from the
extractor. Unscored segments get zero extra credit. All scores, means, scales and
token advantages are frozen before the differentiable loss.

During updates 1–8, the effective token advantage is
`base_advantage + lambda * centered_confidence`. From update 9, the original
prepared batch is returned without token credit. The algorithm, learning rate
schedule and optimizer state continue unchanged. A zero contribution during the
warm start also returns the original batch exactly.

The configurable share target is the ratio of total absolute credit to total
absolute base advantage on eligible tokens in mixed groups. Optional `task_scores`
define mixed groups separately from shaped rewards; without them, batch rewards
define mixed groups. They never enter step confidence. Uniform groups can still
receive credit from a previously calibrated scale.

Calibration targets share `0.875`. A relative p99 cap of `10` uses the median
absolute base advantage in mixed groups. Below four mixed groups carrying credit,
the controller retains its previous scale; with no previous scale it uses an
available first estimate, or zero if there is no signal. Unlike the original
hold path, this implementation reapplies the current measured p99 cap after
holding. Without a measurable denominator, share/p99 are unavailable; the held
scale still applies. The share target is not a hard bound.

## Python integration

Use `GRPOAlgorithm()` for the existing token-mean reduction, or
`GRPOAlgorithm(loss_reduction="sequence_mean")` to average completions equally.
Both support micro-batching. To explicitly compose with score centering using
normalized GRPO rewards, use:

```python
algorithm = ScoreCenteringAlgorithm(
    sampler_top_k=128,
    normalize_rewards=True,
    loss_reduction="sequence_mean",
)
```

That combination also requires `SamplingConfig(record_sampler_top_k=128)` while
collecting rollouts, including after the CONF warm start. Selecting CONF alone
does not turn on sampler capture or score centering.

Inside an existing single-turn training loop:

```python
from mlxrl.confidence import ConfidenceWarmStart
from mlxrl.config import ConfidenceConfig
from mlxrl.data.confidence import arithmetic_steps
from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

confidence = ConfidenceWarmStart(ConfidenceConfig(enabled=True))

for update in range(1, total_updates + 1):
    # Collect fresh completions and their scalar rewards with the current model.
    batch = batch_from_rollouts(
        model, completions, rewards, group_size, pad_token_id, algorithm,
        compute_reference=beta != 0,
        record_confidence=confidence.active(update),
    )
    spans = (
        [arithmetic_steps(tokenizer, c.completion_tokens) for c in completions]
        if confidence.active(update) else []
    )
    batch, credit_metrics = confidence.apply(
        batch, spans, update=update, group_size=group_size,
        task_scores=task_scores,  # optional unshaped task scores
    )
    metrics = optimizer_step(
        model, optimizer, batch, beta, pad_token_id, algorithm,
        micro_batch_size=2,
    )
```

`ConfidenceStep(TokenSpan(start, end), results=(...), exclusions=(...))` uses
half-open indices into completion tokens. Steps must be ordered, non-overlapping
and in bounds. Result and exclusion spans must lie within their step. A custom
`ConfidenceExtractor` can replace `arithmetic_steps`. It should identify step
structure without using ground-truth correctness. Final answers must lie outside
step spans, or be explicitly excluded.

The supplied reader scores the last numeric result on each explicit arithmetic
`step:` line, excluding every written result. It stops at answer markers and
fails closed when decoded token prefixes cannot be aligned. It is intentionally
narrower than RAMP's task-specific variable-chain parser.

Call the controller once for each optimizer update. Save `confidence.state_dict()`
with the model, optimizer, RNG and matching config when implementing checkpoint
resume; restore with `load_state_dict()` and continue at the next update number.
The CLI's NPZ is a diagnostic artifact, not a complete training checkpoint.
Read it with `mlx.core.load` to preserve MLX's bfloat16 arrays.

## Comparison protocol

RAMP uses normalized rewards, per-completion averaging and a score-centered
policy gradient. MLXRL's default GRPO and default score-centering modes differ
from that recipe. Keep normalization, reduction, prompts and sampler settings
fixed when toggling CONF. The supplied example leaves score centering off;
`benchmarks/run_confidence.py` explicitly tests both choices independently.
CONF leaves optimizer settings with the caller. The MLX screening uses a constant
learning rate and no gradient clipping; RAMP's original training also used a
learning-rate warmup and gradient clipping. The screening is a library integration
check with matched controls, rather than a replication of that complete recipe.
Memory preflight uses the existing baseline anchors; those estimates have not
been calibrated for CONF's extra fp32 probability capture.

The short screening uses three seeds, four arms, a pinned local 4-bit model,
two-hop arithmetic and held-out arithmetic examples. Training time includes
rollouts, batch preparation, step extraction, calibration and optimizer updates;
evaluation time is excluded. The script preserves model/source hashes, finite
diagnostics and per-update timing. It refuses to overwrite earlier results.

Correctness tests cover the analytic token-gradient formula, weighted centering,
result/answer exclusion, malformed spans, raw fp32 capture, frozen credit,
uniform-group credit, scale caps, state restoration, exact handoff and padded
float/4-bit micro-batch equivalence. Short arithmetic results do not establish
RAMP's reported approximately 2× time-to-target improvement on MLX.

The [completed 12-run report](../benchmarks/results/confidence_4bit_2026-11-01.md)
includes all three seeds and both base objectives. Results are mixed for GRPO;
the score-centering combination regressed on two seeds and tied on one. CONF
remains an experimental opt-in.
