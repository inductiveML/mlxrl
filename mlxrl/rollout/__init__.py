"""Rollout engines for batched group generation."""

from mlxrl.rollout.naive import Completion, SamplingConfig, generate_group_rollouts
from mlxrl.rollout.optimized import (
    FixedKVCache,
    PrefixCache,
    clone_prompt_cache,
    fixed_decode_cache_from_prefix,
    fixed_decode_cache_from_prefixes,
    generate_group_from_prefix_cache,
    generate_prefix_cached_group_rollouts,
    generate_prompt_set_from_prefix_caches,
    prefill_prompt_once,
)

__all__ = [
    "Completion",
    "FixedKVCache",
    "PrefixCache",
    "SamplingConfig",
    "clone_prompt_cache",
    "fixed_decode_cache_from_prefix",
    "fixed_decode_cache_from_prefixes",
    "generate_group_from_prefix_cache",
    "generate_group_rollouts",
    "generate_prefix_cached_group_rollouts",
    "generate_prompt_set_from_prefix_caches",
    "prefill_prompt_once",
]
