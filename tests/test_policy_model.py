from __future__ import annotations

import mlx.core as mx
import pytest

from mlxrl.policy.model import assert_only_lora_trainable, count_parameters


def test_assert_only_lora_trainable_accepts_adapter_leaves() -> None:
    params = {
        "layers": [
            {
                "self_attn": {
                    "q_proj": {
                        "lora_a": mx.array([1.0, 2.0]),
                        "lora_b": mx.array([0.0]),
                    }
                }
            }
        ]
    }

    paths = assert_only_lora_trainable(params)

    assert paths == (
        "layers.0.self_attn.q_proj.lora_a",
        "layers.0.self_attn.q_proj.lora_b",
    )
    assert count_parameters(params) == 3


def test_assert_only_lora_trainable_rejects_base_weights() -> None:
    params = {"layers": [{"self_attn": {"q_proj": {"weight": mx.array([1.0])}}}]}

    with pytest.raises(ValueError, match="Only LoRA adapter params may be trainable"):
        assert_only_lora_trainable(params)


def test_assert_only_lora_trainable_rejects_empty_tree() -> None:
    with pytest.raises(ValueError, match="No trainable parameters"):
        assert_only_lora_trainable({})
