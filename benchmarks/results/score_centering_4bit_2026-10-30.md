# Experimental score centering: local 4-bit screening

Date: 2026-10-30. Hardware: Apple M4 Max, 48 GiB unified memory.

**Result:** the implementation passes its correctness gates. These short runs do not establish a reliable accuracy or long-run stability advantage, so the objective remains opt-in and experimental.

## Protocol

- Model: `mlx-community/Qwen3-0.6B-4bit`, revision `73e3e38d981303bc594367cd910ea6eb48349da8`; 4-bit weights, quantization group size 64.
- Runtime: mlx 0.32.3, mlx-lm 0.32.0.
- Three seeds: 7, 17, 27. Each run starts a fresh process/model/optimizer.
- Four methods, two sampling conditions, 40 optimizer steps each: **24 runs, 960 steps**.
- Two training prompts per step, four completions per prompt; maximum 24 generated tokens, stopping at the first newline.
- QLoRA rank 8, scale 20, dropout 0; Adam learning rate 1e-5; reference KL coefficient 0.
- Eight fixed training arithmetic cases and eight disjoint held-out cases. Evaluation samples two answers per held-out case, yielding only 16 samples per seed.
- Training reward is `1 - 2 * min(abs(prediction-target)/max(abs(target),1),1)`; malformed answers receive -1. Reported accuracy is strict integer exact match, separate from that shaped reward.
- Unfiltered sampling uses temperature 1 / top-p 1. Default sampling uses temperature 0.7 / top-p 0.95. Both retain top-k=0 and min-p=0.
- PG and SC both capture 128 sampler candidates and use identical centered rewards, fp32 trainer probabilities, and token-mean reduction. Their objective difference is the centering correction.
- Existing GRPO and DAPO retain their original normalization and full-forward old-policy ratios, with capture disabled. DAPO dynamic sampling is disabled. These comparisons are contextual rather than the pure centering ablation.
- Sampling and forward loss are warmed up; first-backward cost remains in timing. Reported speed is generated tokens divided by end-to-end training-loop seconds. Generation lengths differ after learning, so these speeds are descriptive rather than a matched-work speedup.

An initial binary-reward pilot at learning rate 1e-4 was rejected because the first paired seed produced only one informative reward group out of 80. The fixed protocol above was then applied to every final seed and method; no final runs were discarded.

## Results

Accuracy is mean +/- sample standard deviation over three seeds. Memory is maximum MLX allocator peak over the seeds, measured from training through final evaluation; it excludes Python-heap and operating-system overhead.

| Sampling | Method | Held-out exact accuracy | End-to-end tokens/s, mean | Peak MLX GB | Informative reward groups by seed, out of 80 |
| --- | --- | ---: | ---: | ---: | --- |
| unfiltered | pg | 6.2% +/- 10.8 pp | 57.5 | 2.848 | 40, 33, 44 |
| unfiltered | score-centering | 8.3% +/- 9.5 pp | 57.1 | 2.878 | 39, 46, 43 |
| unfiltered | grpo | 8.3% +/- 7.2 pp | 55.3 | 2.599 | 53, 15, 49 |
| unfiltered | dapo | 0.0% +/- 0.0 pp | 52.3 | 2.594 | 56, 37, 30 |
| default | pg | 4.2% +/- 7.2 pp | 34.7 | 2.848 | 23, 43, 22 |
| default | score-centering | 6.2% +/- 6.2 pp | 31.6 | 2.878 | 48, 27, 32 |
| default | grpo | 0.0% +/- 0.0 pp | 42.9 | 2.599 | 20, 43, 17 |
| default | dapo | 2.1% +/- 3.6 pp | 40.3 | 2.594 | 3, 30, 10 |

All 960 steps had finite reported diagnostics. PG and SC also recorded finite gradient norms; stock GRPO/DAPO runs do not collect that optional diagnostic. SC had a smaller peak gradient norm in 4 of six paired PG comparisons, with the opposite result in the remaining pairs.

All twelve PG/SC runs exceeded the predeclared learning-signal check of at least 20 reward-varying groups out of 80. Four stock-baseline runs fell below that threshold; they remain included in the table and raw data:

- unfiltered, seed 17, grpo: 15/80 informative groups.
- default, seed 7, dapo: 3/80 informative groups.
- default, seed 27, grpo: 17/80 informative groups.
- default, seed 27, dapo: 10/80 informative groups.

The evaluation set is small and several methods lose accuracy on some seeds. The observed mean differences are inconclusive. This is one short arithmetic task on one 0.6B model, rather than a reproduction of the paper or evidence about long-horizon or asynchronous stale-policy training.

## Capture overhead and correctness

Twelve alternating-order paired rollout timings gave a median capture overhead of 4.1% for k=128. The paired range was -20.1% to 43.3%, demonstrating substantial timing noise on this short fixture. Every pair produced identical token sequences with capture off and on.

- 170 native tests passed, plus Ruff and Pyright.
- Wheel and source distribution built with the locked Hatchling backend; Twine checks and package-content checks passed.
- Exact enumeration checks covariance gradients, constant-reward drift cancellation, matched-distribution gradients, and the top-k/tail reduction.
- Independent probability arithmetic and 20,000 sampler draws check filtering/temperature, including greedy point masses.
- Tests cover sampled tokens outside the recorded head, exhausted tails, padding, micro-batching, and candidate gradients on both ordinary and 4-bit LoRA models.
- Real Qwen3-0.6B-4bit checks preserved tokens and trace alignment for prefix, batched, and compiled engines.
- Real-model checkpointed, micro-batched score-centering training produced a nonzero finite gradient norm.
- CLI smoke verified persisted selected-token probabilities and 32-candidate heads; these differ from the legacy raw/old-policy logprobs.

## Reproduction

Download the model revision above to a local snapshot and set `MLXRL_SCORE_MODEL` to that directory. From the repository root:

```bash
UV_CACHE_DIR=.uv-cache HF_HUB_OFFLINE=1 uv run python -m benchmarks.run_score_centering \
  --model "$MLXRL_SCORE_MODEL" --steps 40 --seeds 7 17 27 \
  --methods pg score-centering --conditions unfiltered default \
  --output benchmarks/results/reproduced_score_centering.jsonl
UV_CACHE_DIR=.uv-cache HF_HUB_OFFLINE=1 uv run python -m benchmarks.run_score_centering \
  --model "$MLXRL_SCORE_MODEL" --steps 40 --seeds 7 17 27 \
  --methods grpo dapo --conditions unfiltered default \
  --output benchmarks/results/reproduced_existing_baselines.jsonl
UV_CACHE_DIR=.uv-cache HF_HUB_OFFLINE=1 uv run python -m benchmarks.run_score_centering \
  --model "$MLXRL_SCORE_MODEL" --profile-capture \
  --output benchmarks/results/reproduced_capture_profile.jsonl
```

Output paths are created exclusively to prevent accidental replacement of evidence.

## Provenance

The measured runtime is preserved in release 0.3.0 apart from the package version literal. Substituting `__version__ = "0.2.4"` into the released version file reproduces the measured source fingerprint exactly. The script hash describes the final reproduction harness, which adds the stock-baseline modes and input validation around the same recorded protocol.

- Base commit: `5ce24dc6e1982337077762c4247569adf64ffe30`.
- Runtime source SHA-256: `60576153b59bb776e143bff203f2944f26070525eb33ba6fe19f4fc8ca9d2d18`; both training manifests identify the measured implementation before the version-only release bump.
- Release 0.3.0 runtime source SHA-256: `fb06932193e72291001c810738410a987a6d67c111664d24fa135dfcd33e8245`.
- Current reproduction script SHA-256: `15fcfe2534a82e773c0134027e0f55f6d2f5f9065bad8d8276f073fbc5561eed`.
- Model config SHA-256: `15d3ac26c043ae477273ed5802ee0f0b33bb14f18c9d3dd70910c02d906e3f1f`.

- model.safetensors SHA-256: `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`.

Raw artifacts:

- [score_centering_4bit_2026-10-30.jsonl](score_centering_4bit_2026-10-30.jsonl)
- [score_centering_existing_baselines_2026-10-30.jsonl](score_centering_existing_baselines_2026-10-30.jsonl)
- [score_centering_capture_2026-10-30.jsonl](score_centering_capture_2026-10-30.jsonl)
- [score_centering_4bit_2026-10-30.summary.json](score_centering_4bit_2026-10-30.summary.json)

Method reference: [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807), equations 8--10.
