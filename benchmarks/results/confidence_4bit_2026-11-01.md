# Experimental CONF: local 4-bit screening

Date: 2026-11-01. Hardware: Apple M4 Max, 48 GiB unified memory.

**Result:** correctness gates pass, but learning outcomes vary strongly by seed.
CONF remains experimental and disabled by default. These runs do not establish
a reliable quality advantage or RAMP's reported approximately 2× time-to-target gain.

## Protocol

- Model: `mlx-community/Qwen3-0.6B-4bit`, revision
  `73e3e38d981303bc594367cd910ea6eb48349da8`, verified 4-bit weights.
- Runtime: MLX 0.32.3, MLX-LM 0.32.0. Each run starts a fresh model and optimizer process.
- Three paired seeds, 7/17/27; four arms; 16 updates each: **12 runs, 192 updates**.
  The run order is shuffled deterministically and recorded in the manifest.
- Every arm uses group-normalized rewards, per-completion loss averaging,
  Adam at constant 1e-5, LoRA rank 8 / scale 20 / dropout 0, and beta 0.
- Four prompt groups per update, two completions per group, microbatch size 2,
  temperature 1, top-p 1, top-k 0, min-p 0, maximum 64 generated tokens.
  Every completed run generated 8,192 training tokens.
- SC-GRPO explicitly enables score centering with normalized GRPO rewards and
  captures the actual sampler probabilities for 32 candidates. This differs from
  MLXRL's default score-centering reward normalization and loss reduction.
- CONF is active only on updates 1–8. Target share 0.875, relative p99 cap 10,
  hold threshold four mixed groups. The cap is reapplied to held scales when measurable.
- All arms use identical structured two-hop arithmetic prompts. Training uses
  eight fixed cases; greedy held-out evaluation uses eight different cases at
  updates 0, 8 and 16. Every run starts at 50% held-out accuracy.
- Training reward is `1 - 2 * min(abs(prediction-target)/max(abs(target),1),1)`;
  malformed answers receive -1. Accuracy is exact equality of the extracted
  numeric answer. This benchmark uses reward variation to identify mixed groups;
  the CLI instead passes its separate binary correctness scores to calibration.
- Time includes rollouts, raw probability preparation, span extraction,
  calibration and optimizer updates. Model loading and evaluation are excluded;
  first-backward costs and diagnostics are included. No isolated capture-overhead
  or time-to-quality speedup is inferred from these wall times.

## Results

Accuracy is mean ± sample standard deviation across three seeds, not a confidence
interval. Each seed has only eight held-out cases (12.5 percentage-point resolution).

| Method | Accuracy at update 8 | Final accuracy, mean ± sample SD | Training seconds, mean |
| --- | ---: | ---: | ---: |
| GRPO | 50.0% | 41.7% ± 36.1 pp | 94.7 |
| GRPO + CONF | 54.2% | 54.2% ± 50.5 pp | 89.0 |
| SC-GRPO | 37.5% | 45.8% ± 43.9 pp | 100.6 |
| SC-GRPO + CONF | 54.2% | 25.0% ± 25.0 pp | 93.2 |

CONF's final GRPO differences for seeds 7/17/27 are **+62.5, +37.5, -62.5 pp**.
With SC-GRPO they are **-25.0, -37.5, 0.0 pp**. The positive GRPO mean therefore
does not describe a consistent benefit. The score-centering combination ends
below its control on two seeds and ties on the third; its mean also falls from
54.2% at the handoff to 25.0% at update 16.

All final runs, including zero-accuracy outcomes, are retained:

| Method | Seed | Update 8 | Update 16 | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| GRPO | 7 | 0.0% | 0.0% | 109.7 |
| GRPO | 17 | 75.0% | 62.5% | 85.6 |
| GRPO | 27 | 75.0% | 62.5% | 88.9 |
| GRPO + CONF | 7 | 62.5% | 62.5% | 81.8 |
| GRPO + CONF | 17 | 100.0% | 100.0% | 97.6 |
| GRPO + CONF | 27 | 0.0% | 0.0% | 87.7 |
| SC-GRPO | 7 | 0.0% | 50.0% | 85.9 |
| SC-GRPO | 17 | 87.5% | 87.5% | 112.1 |
| SC-GRPO | 27 | 25.0% | 0.0% | 103.8 |
| SC-GRPO + CONF | 7 | 50.0% | 25.0% | 100.2 |
| SC-GRPO + CONF | 17 | 100.0% | 50.0% | 88.1 |
| SC-GRPO + CONF | 27 | 12.5% | 0.0% | 91.4 |

This is a short integration screening on one small model. RAMP's original
variable-chain experiments used a different model/task, larger groups, a
learning-rate warmup and gradient clipping. These MLX runs use constant learning
rate and no gradient clipping. They do not reproduce that complete recipe.
Evaluation at only three checkpoints also does not support a precise
time-to-target comparison. Longer runs, larger evaluation sets and additional
models are needed before claiming a general learning or stability benefit.

## Credit and correctness

| Method | Seed | Updates with credit / 8 | Held updates | Capped updates | Maximum measured relative p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO + CONF | 7 | 8/8 | 4 | 1 | 10.000 |
| GRPO + CONF | 17 | 8/8 | 7 | 0 | 5.604 |
| GRPO + CONF | 27 | 8/8 | 5 | 2 | 10.000 |
| SC-GRPO + CONF | 7 | 7/8 | 5 | 1 | 10.000 |
| SC-GRPO + CONF | 17 | 8/8 | 7 | 0 | 5.604 |
| SC-GRPO + CONF | 27 | 8/8 | 7 | 3 | 10.000 |

Credit was nonzero on 47/48 warm-start updates. The controller held its previous
scale on 35/48 updates and applied the measured cap seven times. These diagnostics
matter: holding can move the realized share away from its target, and the cap is
defined on the measurable mixed-group population, not as a global gradient bound.

- All 192 updates have finite reported diagnostics, and every final adapter is finite.
  Gradient norms are recorded for SC on every update and for GRPO when token credit
  is attached; stock GRPO does not always collect that optional diagnostic.
- Every CONF run switches off at update 9; controls never enable CONF. Source,
  harness and model fingerprints match the recorded experiment.
- Initial recorded first completions match across all four arms for each seed.
- 233 native tests passed, plus Ruff and full Pyright. Coverage includes independent
  analytic GRPO and score-centered token gradients, detached advantages, result and
  answer exclusion, joint multi-token confidence, malformed/unfinished numerics,
  bounded whitespace scanning, calibration/hold/caps, state restoration, exact
  handoff and ragged float/4-bit microbatch equivalence.
- Real Qwen checkpointed/micro-batched CLI runs verify calibrated credit, persistent
  Adam updates, update-nine handoff, saved diagnostics and SC sampler capture.
  See [CLI evidence](confidence_cli_2026-11-01.json). These are integration checks,
  not additional held-out benchmark seeds.

## Pauses and retained evidence

An earlier partial experiment was stopped to fix parser edge cases and is retained
as [prevalidation data](confidence_4bit_2026-11-01.prevalidation.jsonl). It is not
pooled into these results; the comparison parameters were unchanged.

The final experiment was paused at the user's request after ten complete runs,
with GRPO+CONF seed 7 interrupted at update 10. Those partial records are retained
in [the interrupted artifact](confidence_4bit_2026-11-01.interrupted.jsonl). That run
restarted from update 1, then SC-GRPO+CONF seed 27 completed. The restarted run's
first ten training diagnostics match its saved prefix exactly, excluding timings.
The primary result contains each complete run once, with no duplicate partial rows.
The execution break and ordinary host timing variability further limit speed claims.

## Reproduction and provenance

Check out the measured source revision, install its locked dependencies, download
the model revision above, and set `MLXRL_CONF_MODEL` to its local snapshot directory.
From the repository root:

```bash
UV_CACHE_DIR=.uv-cache HF_HUB_OFFLINE=1 uv run python -m benchmarks.run_confidence \
  --model "$MLXRL_CONF_MODEL" --steps 16 --max-tokens 64 --seeds 7 17 27 \
  --methods grpo grpo-conf sc-grpo sc-grpo-conf \
  --output benchmarks/results/reproduced_confidence.jsonl
```

Output paths are created exclusively to preserve earlier experiments.

- Measured source commit: `d4b970e08bc80cf237bd178aa1edd830794fcade`.
- Runtime source SHA-256: `1ef24cd51e9abffcad8c9254b516040bfb1d84a2166288a31e5b1c57c3eb9bfe`.
- Harness SHA-256: `acabda1b3da14362bc08c59f4b27daa2951037f06b44981c293aa121dff6fb67`.
- Model weights SHA-256: `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`.

Final artifacts:

- [Raw final runs](confidence_4bit_2026-11-01.jsonl)
- [Summary and integrity checks](confidence_4bit_2026-11-01.summary.json)
- [CONF API and comparison notes](../../docs/confidence.md)
