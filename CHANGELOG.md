# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning once releases are cut.

## [0.2.4] - 2026-10-22

### Security

- Require MLX-LM 0.32.0 or later with its MLX 0.32.2 minimum, and disable custom model Python
  execution in library, example, and benchmark loaders. The older minimum
  allowed the unsafe `model_file` behavior addressed by upstream's
  [CVE-2026-5843 fix](https://github.com/ml-explore/mlx-lm/pull/1385).
- Parse answer and action tags in linear time to prevent malformed model
  output from stalling reward evaluation and multi-turn rollouts.
- Reject CLI output paths whose symlinks resolve outside the working directory.
- Pin release actions to verified commits, freeze build and validation tools,
  audit all locked dependencies in CI, and require release tags to match a
  package version on main with successful push CI.
- Add security reporting guidance, ownership of release controls, and weekly
  dependency and action updates.

## [0.2.3] - 2026-10-21

### Changed

- Project only completion positions through the output head for stock Qwen2,
  Qwen3, and Llama models, reducing training logit memory while preserving
  logprobs and gradients. Custom subclasses and wrappers retain the full
  forward path. Thanks to @TarangKhanna for PR #6.
- Refresh runtime, development, and benchmark dependencies while preserving
  the seven-day package-age cutoff in the lockfile.
- Include the Pillow, aiohttp, AnyIO, and urllib3 updates from Dependabot
  PRs #1 through #4.

### Fixed

- Support MLX-LM 0.32 prompt-cache cloning while retaining older cache support.
  Thanks to @TarangKhanna for PR #5.
- Use valid prefix lengths when building compiled decode caches from MLX-LM
  0.32 cache buffers, and restore random state through MLX 0.32's mutable key.
- Keep scalar metric conversions compatible with the current MLX type stubs.
- Override the benchmark-only mlx-tune dependency's old datasets constraint
  to use patched datasets 5.x (CVE-2026-66007).

## [0.2.2] - 2026-06-25

### Added

- ECHO world-modeling support for tagged multi-turn trajectories, composing an
  independently normalized ECHO SFT loss with the existing GiGPO action loss.
- `echo_alpha` scheduling with constant and linear-taper-to-zero modes, plus
  separate `loss_echo` and `echo_accuracy` diagnostics.
- Real-Qwen smoke coverage for tagged ECHO batches on Qwen3-0.6B and
  Qwen3.5-9B MLX 4-bit models.

## [0.2.1] - 2026-06-25

### Fixed

- Keep model-backed agentic/GiGPO decode tokens rank-2 during KV-cache decode,
  fixing real Qwen3 model rollouts.
- Use valid repo-relative output paths in the committed Qwen smoke configs.
- Preserve eval-mode old-policy/reference logprob prep while restoring caller
  train mode afterward.

## [0.2.0] - 2026-06-24

### Added

- Multi-turn agentic rollout path with Gym-style `Environment` / `EnvFactory`
  protocols and deterministic reference environments.
- Trajectory data model with action-token spans and full-forward trajectory
  logprob gathering for 4-bit policy, old-policy, and reference semantics.
- GiGPO (`Group-in-Group Policy Optimization`) with episode-level and
  anchor-state step-level advantages, plus an exact `omega=0` reduction gate.
- Trajectory batch builder and optimizer step that preserve the existing
  single-turn GRPO-family APIs.
- Config-driven GiGPO training route with `max_turns`, `rollout_mode`,
  `env_name`, `max_observation_len`, `gigpo_omega`, `gigpo_gamma`, and
  normalization knobs.

### Changed

- Extend the memory estimator to label multi-turn configs as estimated
  OOM-risk shapes while preserving the measured v0.1 single-turn anchors.

## [0.1.2] - 2026-06-24

### Fixed

- Move the Python `<3.15` ceiling from package-level `Requires-Python` to the
  MLX runtime dependency markers so universal resolvers can add the package to
  projects with broad Python ranges while skipping unsupported MLX splits.

## [0.1.1] - 2026-06-24

### Fixed

- Mark `mlx` and `mlx-lm` runtime dependencies as macOS arm64-only and bound the
  package to Python 3.11 through 3.14 so universal resolvers do not try to solve
  unsupported MLX platform or Python splits.

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
