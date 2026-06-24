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

Use `RLOOAlgorithm` in `mlxrl/algo/grpo.py` as the smallest template.

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

## MLX And CI

GitHub-hosted Linux runners cannot run the MLX/Metal path. Linux CI gates the
CPU-safe subset: config validation, memory-estimator math, import-direction
checks, reward functions, and benchmark-result rendering.

Tests marked `@pytest.mark.metal` require Apple Silicon with MLX/Metal. Run the
full suite locally on a Mac before cutting releases. A self-hosted Mac runner
would close this coverage gap.

## Release Cycle

Releases are tag-driven and publish to PyPI through Trusted Publishing. Do not
store PyPI API tokens in GitHub secrets.

One-time PyPI setup:

- create a PyPI Trusted Publisher for project `inductive-mlxrl`;
- set owner to `inductiveML` and repository to `mlxrl`;
- set workflow name to `release.yml`;
- set environment name to `pypi`;
- require manual approval on the GitHub `pypi` environment before publishing.

For each release:

1. Update the version in `pyproject.toml`.
2. Add the release notes to `CHANGELOG.md`.
3. Run the local gates:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pyright
UV_CACHE_DIR=.uv-cache uv build
UV_CACHE_DIR=.uv-cache uv run --no-project --python 3.11 --with twine twine check dist/*
```

4. Commit and merge the release prep.
5. Tag the release from `main`:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The `Release` workflow builds the source distribution and wheel, checks package
metadata with Twine, stores the artifacts, publishes them to PyPI from the
`pypi` environment, and creates a GitHub Release with the built distributions
attached. Running the workflow manually builds and checks artifacts without
publishing because the publish and GitHub Release jobs only run for `v*.*.*`
tags.

The PyPI distribution name is `inductive-mlxrl`; the Python import package and
CLI command remain `mlxrl`.

## Scope

In scope:

- single-process Apple Silicon RL post-training;
- QLoRA on local MLX LLMs;
- critic-free on-policy algorithms;
- memory-conscious rollout and training paths.

Out of scope:

- PPO or other critic/value-model algorithms;
- DPO/ORPO and other offline preference objectives;
- CUDA, torch fallback, or distributed training;
- inference servers or second reference-model copies.
