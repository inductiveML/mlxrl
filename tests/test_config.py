from __future__ import annotations

import pytest

from mlxrl.config import (
    ConfigError,
    TrainConfig,
    estimate_peak_memory_gb,
    memory_fit,
    validate_output_path,
)


def test_train_config_loads_toml_file(tmp_path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        """
model_id = "mlx-community/Qwen3-0.6B-4bit"
quant_bits = 4
group_size = 4
max_completion_len = 256
max_prompt_len = 4096
steps = 1
kl_beta = 0.0
seed = 13

[optimizer]
learning_rate = 0.00001

[algorithm]
name = "grpo"

[sampling]
temperature = 0.0
top_p = 1.0
""".strip()
    )

    config = TrainConfig.from_file(path)

    assert config.model_id == "mlx-community/Qwen3-0.6B-4bit"
    assert config.group_size == 4
    assert config.max_completion_len == 256
    assert config.algorithm.name == "grpo"
    assert config.sampling.temperature == 0.0


def test_train_config_rejects_incoherent_values() -> None:
    with pytest.raises(ConfigError, match="group_size must be at least 1"):
        TrainConfig.from_mapping({"group_size": 0})


def test_train_config_rejects_empty_output() -> None:
    with pytest.raises(ConfigError, match="output must be a non-empty string"):
        TrainConfig.from_mapping({"output": ""})


def test_train_config_rejects_output_path_escape() -> None:
    with pytest.raises(ConfigError, match="relative path inside the working directory"):
        TrainConfig.from_mapping({"output": "../outside.npz"})


def test_train_config_rejects_absolute_output() -> None:
    with pytest.raises(ConfigError, match="relative path inside the working directory"):
        TrainConfig.from_mapping({"output": "/tmp/outside.npz"})


def test_validate_output_path_accepts_nested_relative_path() -> None:
    assert validate_output_path("reference_outputs/phase1_reference.npz").parts == (
        "reference_outputs",
        "phase1_reference.npz",
    )


def test_memory_fit_flags_large_9b_config_and_suggests_fallback() -> None:
    config = TrainConfig(
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        group_size=8,
        max_prompt_len=97,
        max_completion_len=512,
        gradient_checkpointing=False,
    )

    estimate = memory_fit(config, available_unified_memory_gb=48.0)

    assert not estimate.fits
    assert estimate.estimated_peak_gb > 48.0
    assert estimate.estimated_peak_gb < 150.0
    assert estimate.suggested_peak_gb <= 48.0
    assert estimate.suggested_config.gradient_checkpointing
    assert estimate.suggested_config.group_size == 4
    assert estimate.suggested_config.max_completion_len == 512
    assert estimate.confidence == "estimated — OOM risk, not measured"
    assert estimate.reduction_hint == "enable gradient_checkpointing"
    assert "OOM risk" in estimate.display_label()
    assert "enable gradient_checkpointing" in estimate.suggestions


def test_memory_fit_known_good_9b_checkpoint_anchor_fits() -> None:
    config = TrainConfig(
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        group_size=2,
        max_prompt_len=97,
        max_completion_len=512,
        gradient_checkpointing=True,
    )

    estimate = memory_fit(config, available_unified_memory_gb=48.0)

    assert estimate.fits
    assert 24.0 <= estimate.estimated_peak_gb <= 28.0


def test_memory_estimator_matches_measured_anchors() -> None:
    standard_06b = TrainConfig(
        model_id="mlx-community/Qwen3-0.6B-4bit",
        group_size=4,
        max_prompt_len=19,
        max_completion_len=256,
    )
    qwen35_g4_checkpoint = TrainConfig(
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        group_size=4,
        max_prompt_len=97,
        max_completion_len=512,
        gradient_checkpointing=True,
    )
    qwen35_short_no_checkpoint = TrainConfig(
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        group_size=2,
        max_prompt_len=48,
        max_completion_len=80,
        gradient_checkpointing=False,
    )

    assert abs(estimate_peak_memory_gb(standard_06b) - 6.2) < 0.1
    assert abs(estimate_peak_memory_gb(qwen35_g4_checkpoint) - 45.9) < 0.1
    assert abs(estimate_peak_memory_gb(qwen35_short_no_checkpoint) - 36.0) < 0.1
