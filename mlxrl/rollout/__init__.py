"""Rollout engines for batched group generation."""

from mlxrl.rollout.naive import Completion, SamplingConfig, generate_group_rollouts
from mlxrl.rollout.optimized import (
    PrefixCache,
    clone_prompt_cache,
    generate_prefix_cached_group_rollouts,
    prefill_prompt_once,
)

__all__ = [
    "Completion",
    "PrefixCache",
    "SamplingConfig",
    "clone_prompt_cache",
    "generate_group_rollouts",
    "generate_prefix_cached_group_rollouts",
    "prefill_prompt_once",
]
