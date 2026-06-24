# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning once releases are cut.

## [0.1.0] - 2026-06-24

### Added

- Single-process MLX GRPO+QLoRA training path for Apple Silicon.
- Prefix-cached grouped rollout engine with compiled decode support.
- Adapter-disabled reference logprob path on one model object.
- Full-forward old-policy logprob recompute for 4-bit correctness.
- Algorithm protocol with GRPO, Dr. GRPO, DAPO, GSPO, and RLOO.
- DAPO `filter_batch` hook for dynamic zero-advantage group filtering.
- Per-layer MLX-LM gradient checkpointing for DeltaNet/linear-attention models.
- Micro-batched gradient accumulation for token-mean losses.
- Typed config schema and memory preflight helper calibrated to measured anchors.
- Phase 4 benchmark harness for `mlxrl`, `mlx-lm`, `mlx-tune`, and `mlx-lm-lora`.
- Linux CPU-safe CI for docs/config/import-direction checks.

### Notes

- `mlxrl` is pre-1.0; public APIs may change while the correctness gates settle.
- Full MLX/Metal tests require Apple Silicon or a self-hosted Mac runner.
- Published on PyPI as `inductive-mlxrl`; the import package and CLI command
  remain `mlxrl`.
