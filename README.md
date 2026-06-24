# mlxrl

Fast on-policy MLX RL for Apple Silicon; not a general RL framework, not
preference tuning, and not distributed training.

`mlxrl` is a small, single-process RL post-training library for LLMs on Apple
Silicon. It is built around one idea: GRPO on MLX should be a fast batched
rollout path with a thin loss and optimizer step on top, not a framework.

The current implementation targets QLoRA GRPO on local 4-bit MLX models. It
reuses `mlx-lm` model loading, LoRA layers, KV caches, and sampling utilities,
and keeps generation and training in one Python process with one model object.

`mlxrl` is pre-1.0. The correctness gates are stable, but import APIs and config
fields may change before a 1.0 release.

## Quickstart

```bash
git clone https://github.com/inductiveML/mlxrl.git
cd mlxrl
UV_CACHE_DIR=.uv-cache uv sync --all-groups
UV_CACHE_DIR=.uv-cache uv run mlxrl train \
  --config examples/qwen3_0_6b_grpo.toml \
  --available-memory-gb 48
```

For the measured 9B-on-48GB shape, use the checkpointed G=2 config:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl train \
  --config examples/qwen35_9b_g2_checkpoint.toml \
  --available-memory-gb 48 \
  --dry-run
```

## What Works

- Batched group rollouts with MLX-LM KV caches and sampling.
- Full-forward old-policy logprob recompute for training-time `pi_old`.
- Adapter-disabled reference policy on the same model object.
- GRPO, Dr. GRPO, DAPO, and GSPO loss variants.
- RLOO (REINFORCE Leave-One-Out) as a critic-free rollout objective.
- QLoRA injection on dense and heterogeneous/hybrid layer stacks.
- Qwen3.5-style hybrid support via MLX-LM auto LoRA targeting, including
  DeltaNet `linear_attn.in_proj_*` and dense attention `q/k/v/o_proj`.
- Per-layer gradient checkpointing through `mlx_lm.tuner.trainer.grad_checkpoint`
  for linear-attention/DeltaNet backward memory.
- Micro-batched gradient accumulation for token-mean policy losses.
- `beta == 0` reference-forward skip.
- Phase 4 benchmark harness for `mlxrl`, `mlx-tune`, `mlx-lm-lora`, and `mlx-lm`.

## Install

After the first tagged release, install the PyPI distribution:

```bash
pip install inductive-mlxrl
```

The Python import package and CLI command are still `mlxrl`:

```bash
mlxrl --help
```

Source installs are also supported:

```bash
UV_CACHE_DIR=.uv-cache uv sync --all-groups
```

Run commands through the local environment:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl --help
```

Python 3.11+ is required. Runtime dependencies are intentionally small:
`mlx` and `mlx-lm`. Development dependencies include `pytest`, `ruff`,
`pyright`, `mlx-tune`, and `mlx-lm-lora` for comparison benchmarks.
The PyPI distribution name is `inductive-mlxrl`; the import package and console
script remain `mlxrl`.

## Quick Smoke Tests

Dense Qwen3 0.6B:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase0-smoke \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "What is 2+2?"
```

Hybrid Qwen3.5 9B with rank-16 LoRA:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase0-smoke \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --rank 16 \
  --scale 2.0 \
  --prompt "What is 2+2?"
```

The smoke gate prints the model id, layer count, LoRA target keys, per-layer
LoRA module counts, total/trainable parameter counts, and logits shape. It
fails if any trainable leaf is not `lora_a` or `lora_b`.

## Training Commands

Toy hand-computed GRPO math gate:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase1-toy-gate
```

Small built-in GSM8K-style run:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase1-gsm8k \
  --model mlx-community/Qwen3-0.6B-4bit \
  --steps 20 \
  --group-size 4 \
  --max-tokens 64
```

Config-driven run:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl train \
  --config examples/qwen3_0_6b_grpo.toml \
  --available-memory-gb 48
```

The config schema validates model id, quant bits, group size, completion/prompt
lengths, checkpointing granularity, `iogpu.wired_limit_mb`, optimizer settings,
algorithm hyperparameters, KL beta, and seed before a model is loaded. CLI
overrides such as `--steps`, `--group-size`, `--max-tokens`, `--algorithm`,
`--beta`, and `--seed` apply on top of the file.

For DeltaNet / linear-attention models, enable per-layer checkpointing:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase1-gsm8k \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --rank 16 \
  --scale 2.0 \
  --checkpoint-completion-forward \
  --steps 1 \
  --group-size 2 \
  --max-tokens 256
```

Despite the historical CLI name, `--checkpoint-completion-forward` now enables
per-transformer-block checkpointing at model setup. The old whole-model
`mx.checkpoint(...)` wrapper was removed because it does not cap DeltaNet's
per-layer scan memory.

Phase 2 rollout equivalence check:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase2-equivalence \
  --model mlx-community/Qwen3-0.6B-4bit \
  --group-size 4 \
  --max-tokens 32 \
  --compile-decode-step \
  --batch-groups
```

## Import API

Minimal model setup:

```python
from mlxrl.policy import LoRAConfig, load_policy_with_lora

model, tokenizer, report = load_policy_with_lora(
    model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
    config=LoRAConfig(
        rank=16,
        scale=2.0,
        dropout=0.0,
        grad_checkpoint=True,
    ),
)
```

One optimizer step:

```python
import mlx.optimizers as optim

from mlxrl.algo import GRPOAlgorithm
from mlxrl.train import batch_from_rollouts, optimizer_step

optimizer = optim.Adam(learning_rate=1e-5)
algorithm = GRPOAlgorithm()
batch = batch_from_rollouts(
    model=model,
    completions=completions,
    rewards=rewards,
    group_size=4,
    pad_token_id=pad_token_id,
    algorithm=algorithm,
    compute_reference=beta != 0.0,
)
metrics = optimizer_step(
    model=model,
    optimizer=optimizer,
    batch=batch,
    beta=beta,
    pad_token_id=pad_token_id,
    algorithm=algorithm,
    use_checkpoint=True,
    micro_batch_size=2,
)
```

`micro_batch_size=0` keeps the original whole-batch path. Micro-batching is
currently exact for token-mean policy losses: base GRPO, DAPO, GSPO token mode,
RLOO, and Dr. GRPO with `loss_reduction="token_mean"`. Sequence-reduced losses
should keep `micro_batch_size=0`.

## Policy Semantics

- The base model is frozen before LoRA injection.
- Only LoRA adapter leaves are trainable.
- Reference logprobs are computed by temporarily disabling adapters on the same
  model object; there is no second reference model in memory.
- Old-policy logprobs are recomputed with a full forward for the training batch.
  Rollout-time logprobs are captured for inspection, but 4-bit sequential decode
  and full-forward prefill are not numerically identical on hybrid/quantized
  models, so recompute remains the default training semantics.
- When `beta == 0`, the reference forward is skipped and the policy logprobs are
  used as a zero-KL placeholder.
- PPO, DPO, and ORPO are intentionally out of scope. PPO needs a separate critic
  and value forward; DPO/ORPO are offline preference objectives with no rollout
  phase. `mlxrl` is critic-free, on-policy, and rollout-based by design.

## Algorithms

Concrete algorithms implement the small `Algorithm` protocol: compute
advantages, optionally filter a prepared batch, then compute a loss from policy,
old-policy, and reference logprobs. `rollout/`, `policy/`, and `train/` do not
import concrete algorithm implementations.

| algorithm | defining behavior |
| --- | --- |
| GRPO | group-normalized rewards, token-level importance ratio |
| Dr. GRPO | centered or normalized rewards with decoupled length reduction |
| DAPO | asymmetric low/high clipping plus optional dynamic zero-advantage group filtering |
| GSPO | sequence-level, length-normalized importance ratio and clipping |
| RLOO | leave-one-out group baseline, no critic, no std-normalized advantage |

## Memory Preflight

`mlxrl train` can estimate memory before loading the model:

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl train \
  --config examples/qwen3_0_6b_grpo.toml \
  --available-memory-gb 48 \
  --dry-run
```

The estimator is calibrated to measured anchors: `6.245 GB` for
Qwen3-0.6B/G4/prompt≈19/T256, `25.9 GB` for
Qwen3.5-9B/G2/seq609/per-layer-checkpointed, `45.9 GB` for
Qwen3.5-9B/G4/seq609/per-layer-checkpointed, and `36 GB` for
Qwen3.5-9B/G2/seq128/no-checkpoint. For hybrid 9B no-checkpoint long-sequence
configs, it reports an OOM-risk lower bound rather than a fake precise peak.
For an obviously too-large Qwen3.5-9B/G8/prompt97/T512/no-checkpoint config on
48 GB, it flags the run and suggests the measured-boundary fallback around
G4/T512/checkpointed.

## Benchmarks

Local M4 Max Phase 4 snapshot:

- `454` rollout tok/s on Qwen3-0.6B GRPO with G=4 and 256-token completions.
- `0.283` end-to-end it/s with full `mlxrl` training semantics.
- `3.2x` faster rollout and `2.2x` higher end-to-end it/s than `mlx-tune`
  v0.5.1 on the same run shape.
- `1.3x` faster rollout than sequential `mlx-lm` generation at G=4.

These are the two-pass means from
`benchmarks/results/gate5_full_reconciled.md`, run with MLX 0.31.2,
MLX-LM 0.31.3, `mlx-community/Qwen3-0.6B-4bit`, 100 measured steps with
5 warmup steps discarded:

| target | comparison | rollout tok/s | grad s/step | samples/s | it/s | peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mlxrl` | apples-to-apples GRPO | 454.1 | 1.282 | 1.133 | 0.283 | 6.25 |
| `mlx-lm` | generation-only, G=1 | 347.0 | - | 1.355 | - | 0.52 |
| `mlx-lm-g4` | generation-only, sequential G=4 | 349.7 | - | 1.366 | - | 0.52 |
| `mlx-tune` | package-speed reference | 142.2 | 0.502 | 0.519 | 0.130 | 6.16 |
| `mlx-lm-lora` | package-speed reference | 557.9 | 0.592 | 1.648 | 0.412 | 5.32 |

`mlx-lm-lora` reports higher raw package-speed throughput in this snapshot, but
its benchmarked path is not the same training problem as `mlxrl`'s live
old-policy/reference semantics and completion-loss masking. That is the honest
case where `mlxrl` is not faster; the apples-to-apples comparison label is
reserved for `mlxrl`'s own semantic path. On the 9B Noether real workload, the
checkpointed MLX path measured about 6x faster than the previous torch-MPS path;
that workload is separate from the public Phase 4 package-speed harness.

Run the Phase 4 harness:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/run_phase4.py run \
  --targets mlxrl,mlx-lm,mlx-tune,mlx-lm-lora \
  --model mlx-community/Qwen3-0.6B-4bit \
  --steps 100 \
  --warmup-steps 5 \
  --group-size 4 \
  --max-tokens 256 \
  --passes 2 \
  --output benchmarks/results/phase4.jsonl \
  --summary benchmarks/results/phase4.md \
  --allow-missing-baselines
```

The harness reports synchronized rollout tok/s, gradient seconds per step,
samples/s, it/s, and peak MLX memory. `mlx-lm` targets are generation-only;
external package targets are useful speed references but may not match `mlxrl`
training semantics.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) and [DESIGN.md](DESIGN.md) before adding
algorithms or changing rollout/logprob semantics.

Run the quality gates:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pyright
```

MLX lazy evaluation matters. Any `mx.eval(...)` or `mx.synchronize()` in this
repo should mark a real boundary: sampled token append/EOS checks, logprob
freezing before adapter mutation, per-micro-batch graph release, optimizer
updates, or benchmark timing boundaries.

## Layout

```text
mlxrl/
  rollout/   # batched group generation
  policy/    # model loading, LoRA setup, logprob passes
  algo/      # GRPO-family advantages and losses
  train/     # value_and_grad and optimizer integration
  data/      # toy GSM8K data and rewards
  cli.py
tests/
benchmarks/
```

## Non-Goals

- No inference server or second model copy.
- No CUDA or torch fallback.
- No distributed training.
- No broad RL framework abstractions beyond the small algorithm interface.
