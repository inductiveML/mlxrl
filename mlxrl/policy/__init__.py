"""Policy model loading, adapter injection, and logprob utilities."""

from mlxrl.policy.logprobs import (
    CompletionLogprobs,
    DualLogprobs,
    adapters_disabled,
    completion_logprobs,
    dual_logprobs,
    pad_token_id_from_tokenizer,
    prefix_cached_completion_logprobs,
)
from mlxrl.policy.model import (
    DEFAULT_LORA_TARGET_SUFFIXES,
    DEFAULT_MODEL_ID,
    LoRAConfig,
    Phase0Report,
    assert_lora_on_every_layer,
    assert_only_lora_trainable,
    enable_grad_checkpointing,
    load_policy_with_lora,
    strict_lora_config,
)

__all__ = [
    "DEFAULT_LORA_TARGET_SUFFIXES",
    "DEFAULT_MODEL_ID",
    "CompletionLogprobs",
    "DualLogprobs",
    "LoRAConfig",
    "Phase0Report",
    "adapters_disabled",
    "assert_lora_on_every_layer",
    "assert_only_lora_trainable",
    "completion_logprobs",
    "dual_logprobs",
    "enable_grad_checkpointing",
    "load_policy_with_lora",
    "pad_token_id_from_tokenizer",
    "prefix_cached_completion_logprobs",
    "strict_lora_config",
]
