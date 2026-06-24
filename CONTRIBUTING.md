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
