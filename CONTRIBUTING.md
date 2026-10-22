# Contributing

`mlxrl` is fast on-policy MLX RL for Apple Silicon. It is not a broad RL
framework, not preference tuning, and not multi-GPU or distributed training.

## Development Loop

Use `uv` from the repository root:

```bash
UV_CACHE_DIR=.uv-cache uv sync --all-groups
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pyright
```

Tests must stay green while refactoring. Correctness tests are not optional:
rollout equivalence, loss/gradient checks, import-direction guards, and memory
estimator anchor tests are part of the contract.

## Adding An Algorithm

Use `RLOOAlgorithm` in `mlxrl/algo/grpo.py` as the smallest single-turn
template. Use `GiGPOAlgorithm` in `mlxrl/algo/gigpo.py` as the trajectory
template.

Required steps:

- implement the `Algorithm` protocol;
- add a hand-computed toy loss and gradient test;
- add a reduction or relationship test when the algorithm should collapse to an
  existing objective under a degenerate config;
- keep algorithm code in `mlxrl/algo/`;
- do not import concrete algorithms from `rollout/`, `policy/`, or `train/`;
- add an end-to-end smoke before relying on a new objective.

If an algorithm needs to drop or reshape examples before the loss, use the
`filter_batch` hook. Do not special-case it in the trainer.

Trajectory algorithms implement `TrajectoryAlgorithm`, ship a hand-computed
multi-turn loss/gradient test, and prove their reduction to an existing
objective when a degenerate config should do so. They may consume trajectory
metadata such as env state ids, but `rollout/`, `policy/`, and `train/` still
must not import concrete algorithms.

## Adding An Environment

External envs integrate through `EnvFactory(task, seed, group_index) ->
Environment`. Implement `reset()`, `step(action)`, `max_turns`, and
`state_id(observation)`. The env owns the anchor-state definition; the rollout
engine only records ids, and GiGPO groups them later.

Keep envs deterministic under the provided seed when possible. Add a tiny test
that resets, steps, returns stable state ids, and exercises truncation or
termination. Do not add omega gym itself to this repo.

## MLX And CI

GitHub-hosted Linux runners cannot run the MLX/Metal path. Linux CI gates the
CPU-safe subset: config validation, memory-estimator math, import-direction
checks, reward functions, and benchmark-result rendering.

Tests marked `@pytest.mark.metal` require Apple Silicon with MLX/Metal. Run the
full suite locally on a Mac before cutting releases. A self-hosted Mac runner
would close this coverage gap.

## Package Compatibility

The PyPI distribution requires Python 3.11+. The MLX runtime path is supported
on Apple Silicon with Python 3.11 through 3.14. Keep the `mlx` and `mlx-lm`
runtime dependencies marked for macOS arm64 and Python `<3.15` so universal
resolvers do not try to solve MLX wheels for unsupported platform or Python
splits.

The development environment overrides mlx-tune 0.6's `datasets<4` constraint
with `datasets>=5.0.1,<6` to avoid CVE-2026-66007. This only affects the external
benchmark tools; the published package depends on MLX and MLX-LM. Remove the
override when mlx-tune accepts a patched datasets release.

Certifi and Pillow use a one-day package-age cutoff so their October 2026
security patches can be installed promptly. Other dependencies retain the
configured default cutoff.

## Release Cycle

Releases are tag-driven and publish to PyPI through Trusted Publishing. Do not
store PyPI API tokens in GitHub secrets.

One-time PyPI setup:

- create a PyPI Trusted Publisher for project `inductive-mlxrl`;
- set owner to `inductiveML` and repository to `mlxrl`;
- set workflow name to `release.yml`;
- set environment name to `pypi`;
- require manual approval on the GitHub `pypi` environment before publishing.

Restrict the environment to tags matching `v*`, disable administrator bypass
for environment approval, and protect release-tag creation, updates, and
deletion so only repository administrators can perform them. Require the
`Linux CPU-safe checks` status on main and retain pull-request review.

The publisher runs the official PyPA container by digest. When updating it,
resolve the upstream release's commit-tagged image from
`ghcr.io/pypa/gh-action-pypi-publish`, verify its content digest and source
metadata, and update the digest and source-commit comment together. Other
actions use commit pins maintained by Dependabot.

For each release:

1. Update the version in `pyproject.toml` and `mlxrl/__init__.py`, then run `uv lock`.
2. Add the release notes to `CHANGELOG.md`.
3. Run the local gates:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --all-groups
UV_CACHE_DIR=.uv-cache uv run --no-sync pytest
UV_CACHE_DIR=.uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=.uv-cache uv run --no-sync pyright
UV_CACHE_DIR=.uv-cache uv run --no-sync python .github/scripts/audit_dependencies.py
UV_CACHE_DIR=.uv-cache uv build --no-build-isolation --python .venv/bin/python
UV_CACHE_DIR=.uv-cache uv run --no-sync twine check dist/*
```

4. Commit and merge the release prep, then wait for main's push CI to pass.
5. Tag the release from `main`:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The `Release` workflow first checks that the tag matches the package version,
identifies a commit already on main, and has successful main push CI. It builds
with frozen validation and build tools, checks metadata with Twine, stores the
artifacts, and publishes signed distributions from the protected `pypi`
environment after maintainer approval. The GitHub Release receives the same
files. Manual runs on branches only build and check artifacts; manual runs on
version tags must satisfy all publication controls.

The PyPI distribution name is `inductive-mlxrl`; the Python import package and
CLI command remain `mlxrl`.

## Scope

In scope:

- single-process Apple Silicon RL post-training;
- QLoRA on local MLX LLMs;
- critic-free on-policy algorithms;
- multi-turn agentic environments through the small env protocol;
- memory-conscious rollout and training paths.

Out of scope:

- PPO or other critic/value-model algorithms;
- DPO/ORPO and other offline preference objectives;
- CUDA, torch fallback, or distributed training;
- inference servers or second reference-model copies.
