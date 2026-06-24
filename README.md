# mlxrl

`mlxrl` is a small, single-process RL post-training library for LLMs on Apple
Silicon. It is built around one idea: GRPO on MLX should be a fast batched
rollout path with a thin loss and optimizer step on top, not a framework.

The current implementation targets QLoRA GRPO on local 4-bit MLX models. It
reuses `mlx-lm` model loading, LoRA layers, KV caches, and sampling utilities,
and keeps generation and training in one Python process with one model object.

## What Works

- Batched group rollouts with MLX-LM KV caches and sampling.
- Full-forward old-policy logprob recompute for training-time `pi_old`.
- Adapter-disabled reference policy on the same model object.
- GRPO, Dr. GRPO, DAPO, and GSPO loss variants.
- QLoRA injection on dense and heterogeneous/hybrid layer stacks.
- Qwen3.5-style hybrid support via MLX-LM auto LoRA targeting, including
  DeltaNet `linear_attn.in_proj_*` and dense attention `q/k/v/o_proj`.
- Per-layer gradient checkpointing through `mlx_lm.tuner.trainer.grad_checkpoint`
  for linear-attention/DeltaNet backward memory.
- Micro-batched gradient accumulation for token-mean policy losses.
- `beta == 0` reference-forward skip.
- Phase 4 benchmark harness for `mlxrl`, `mlx-tune`, `mlx-lm-lora`, and `mlx-lm`.

## Install

Use `uv` from the repository root:

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

from mlxrl.train import batch_from_rollouts, optimizer_step

optimizer = optim.Adam(learning_rate=1e-5)
batch = batch_from_rollouts(
    model=model,
    completions=completions,
    rewards=rewards,
    group_size=4,
    pad_token_id=pad_token_id,
    compute_reference=beta != 0.0,
)
metrics = optimizer_step(
    model=model,
    optimizer=optimizer,
    batch=batch,
    beta=beta,
    pad_token_id=pad_token_id,
    use_checkpoint=True,
    micro_batch_size=2,
)
```

`micro_batch_size=0` keeps the original whole-batch path. Micro-batching is
currently exact for token-mean policy losses: base GRPO, DAPO, GSPO token mode,
and Dr. GRPO with `loss_reduction="token_mean"`. Sequence-reduced losses should
keep `micro_batch_size=0`.

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

## Benchmarks

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
