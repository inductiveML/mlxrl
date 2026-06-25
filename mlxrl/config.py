"""Typed training config and memory-fit estimates for mlxrl."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from mlxrl.echo import EchoSchedule, EchoScheduleName


class ConfigError(ValueError):
    """Raised when a training config is invalid."""


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-5

    def validate(self) -> None:
        if self.name != "adam":
            raise ConfigError("optimizer.name must be 'adam'.")
        if self.learning_rate <= 0:
            raise ConfigError("optimizer.learning_rate must be positive.")


@dataclass(frozen=True)
class SamplingConfigData:
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 0

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigError("sampling.temperature must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ConfigError("sampling.top_p must be in (0, 1].")
        if self.min_p < 0:
            raise ConfigError("sampling.min_p must be non-negative.")
        if self.top_k < 0:
            raise ConfigError("sampling.top_k must be non-negative.")


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str = "grpo"
    clip_low: float | None = None
    clip_high: float | None = None
    dapo_dynamic_sampling: bool = False
    gspo_importance: str = "sequence"
    drgrpo_normalize_rewards: bool = False
    drgrpo_loss_reduction: str = "sequence_max_tokens"
    drgrpo_max_tokens: int | None = None
    gigpo_omega: float = 1.0
    gigpo_gamma: float = 1.0
    gigpo_normalization: str = "std"

    def validate(self) -> None:
        normalized = normalize_algorithm_name(self.name)
        if normalized not in {"grpo", "dr-grpo", "dapo", "gspo", "rloo", "gigpo"}:
            raise ConfigError(
                "algorithm.name must be one of: grpo, dr-grpo, dapo, gspo, rloo, gigpo."
            )
        if self.clip_low is not None and self.clip_low < 0:
            raise ConfigError("algorithm.clip_low must be non-negative or null.")
        if self.clip_high is not None and self.clip_high < 0:
            raise ConfigError("algorithm.clip_high must be non-negative or null.")
        if self.gspo_importance not in {"sequence", "token"}:
            raise ConfigError("algorithm.gspo_importance must be 'sequence' or 'token'.")
        if self.drgrpo_loss_reduction not in {"sequence_max_tokens", "token_mean"}:
            raise ConfigError(
                "algorithm.drgrpo_loss_reduction must be "
                "'sequence_max_tokens' or 'token_mean'."
            )
        if self.drgrpo_max_tokens is not None and self.drgrpo_max_tokens <= 0:
            raise ConfigError("algorithm.drgrpo_max_tokens must be positive or null.")
        if self.gigpo_omega < 0:
            raise ConfigError("algorithm.gigpo_omega must be non-negative.")
        if self.gigpo_gamma < 0:
            raise ConfigError("algorithm.gigpo_gamma must be non-negative.")
        if self.gigpo_normalization not in {"std", "center"}:
            raise ConfigError("algorithm.gigpo_normalization must be 'std' or 'center'.")


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = "mlx-community/Qwen3-0.6B-4bit"
    quant_bits: int = 4
    group_size: int = 4
    max_completion_len: int = 256
    max_prompt_len: int = 4096
    max_turns: int = 1
    max_observation_len: int = 0
    rollout_mode: str = "parallel_per_turn"
    env_name: str = "single-turn"
    steps: int = 20
    seed: int = 7
    kl_beta: float = 0.04
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    checkpoint_granularity: str = "per-layer"
    iogpu_wired_limit_mb: int | None = None
    micro_batch_size: int = 0
    use_chat_template: bool = False
    echo_alpha: float = 0.0
    echo_schedule: str = "constant"
    echo_taper_steps: int | None = None
    output: str = "reference_outputs/phase1_reference.npz"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampling: SamplingConfigData = field(default_factory=SamplingConfigData)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def validate(self) -> None:
        if not self.model_id:
            raise ConfigError("model_id must be non-empty.")
        if self.quant_bits not in {4, 8, 16}:
            raise ConfigError("quant_bits must be one of: 4, 8, 16.")
        if self.group_size < 1:
            raise ConfigError("group_size must be at least 1.")
        if self.max_completion_len < 1:
            raise ConfigError("max_completion_len must be at least 1.")
        if self.max_prompt_len < 1:
            raise ConfigError("max_prompt_len must be at least 1.")
        if self.max_turns < 1:
            raise ConfigError("max_turns must be at least 1.")
        if self.max_observation_len < 0:
            raise ConfigError("max_observation_len must be non-negative.")
        if self.rollout_mode not in {"parallel_per_turn", "sequential"}:
            raise ConfigError("rollout_mode must be 'parallel_per_turn' or 'sequential'.")
        if not self.env_name:
            raise ConfigError("env_name must be non-empty.")
        if self.steps < 1:
            raise ConfigError("steps must be at least 1.")
        if self.kl_beta < 0:
            raise ConfigError("kl_beta must be non-negative.")
        if self.rank < 1:
            raise ConfigError("rank must be at least 1.")
        if self.scale <= 0:
            raise ConfigError("scale must be positive.")
        if not 0 <= self.dropout < 1:
            raise ConfigError("dropout must be in [0, 1).")
        if self.checkpoint_granularity != "per-layer":
            raise ConfigError("checkpoint_granularity must be 'per-layer'.")
        if self.iogpu_wired_limit_mb is not None and self.iogpu_wired_limit_mb <= 0:
            raise ConfigError("iogpu_wired_limit_mb must be positive or null.")
        if self.micro_batch_size < 0:
            raise ConfigError("micro_batch_size must be non-negative.")
        if self.echo_alpha < 0:
            raise ConfigError("echo_alpha must be non-negative.")
        try:
            EchoSchedule(
                alpha=self.echo_alpha,
                schedule=cast(EchoScheduleName, self.echo_schedule),
                taper_steps=self.echo_taper_steps,
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        validate_output_path(self.output)
        self.optimizer.validate()
        self.sampling.validate()
        self.algorithm.validate()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TrainConfig:
        try:
            optimizer = OptimizerConfig(**dict(raw.get("optimizer", {})))
            sampling = SamplingConfigData(**dict(raw.get("sampling", {})))
            algorithm = AlgorithmConfig(**dict(raw.get("algorithm", {})))
            top_level = {
                key: value
                for key, value in raw.items()
                if key not in {"optimizer", "sampling", "algorithm"}
            }
            config = cls(
                **top_level,
                optimizer=optimizer,
                sampling=sampling,
                algorithm=algorithm,
            )
        except TypeError as error:
            raise ConfigError(f"invalid config field: {error}") from error
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> TrainConfig:
        config_path = Path(path)
        if config_path.suffix == ".json":
            raw = json.loads(config_path.read_text())
        else:
            raw = tomllib.loads(config_path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError("config file must contain a mapping/object.")
        return cls.from_mapping(raw)


@dataclass(frozen=True)
class MemoryEstimate:
    estimated_peak_gb: float
    available_gb: float
    fits: bool
    suggested_config: TrainConfig
    suggested_peak_gb: float
    suggestions: tuple[str, ...]
    confidence: str
    reduction_hint: str | None = None

    @property
    def warning(self) -> str | None:
        if self.fits:
            return None
        return (
            f"estimated peak {self.estimated_peak_gb:.1f} GB exceeds available "
            f"{self.available_gb:.1f} GB ({self.confidence}); suggested fallback estimates "
            f"{self.suggested_peak_gb:.1f} GB"
        )

    def display_label(self) -> str:
        """User-facing confidence label for CLI output."""

        if self.reduction_hint is None:
            return self.confidence
        return f"{self.confidence}; reduce: {self.reduction_hint}"


def normalize_algorithm_name(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized == "drgrpo":
        return "dr-grpo"
    if normalized in {"gig-po", "gigpo"}:
        return "gigpo"
    return normalized


def validate_output_path(value: str, field_name: str = "output") -> Path:
    """Validate an output path that the CLI may create before writing artifacts."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")
    output_path = Path(value.strip())
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ConfigError(f"{field_name} must be a relative path inside the working directory.")
    return output_path


def model_size_billions(model_id: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_id, flags=re.IGNORECASE)
    if match is None:
        return 0.6
    return float(match.group(1))


def effective_sequence_length(config: TrainConfig) -> int:
    """Return the sequence length used by the memory preflight estimate."""

    action_tokens = config.max_turns * config.max_completion_len
    observation_tokens = max(0, config.max_turns - 1) * config.max_observation_len
    return config.max_prompt_len + action_tokens + observation_tokens


def estimate_peak_memory_gb(config: TrainConfig) -> float:
    """Estimate peak unified memory from empirical mlxrl anchors.

    Anchors:
    - Qwen3-0.6B-4bit, G=4, prompt≈19, T=256: 6.245 GB.
    - Qwen3.5-9B-4bit, G=2, seq=609, per-layer checkpointing: 25.9 GB.
    - Qwen3.5-9B-4bit, G=4, seq=609, per-layer checkpointing: 45.9 GB.
    - Qwen3.5-9B-4bit, G=2, seq=128, no checkpointing: 36 GB.

    Long-sequence uncheckpointed hybrid estimates are OOM-risk lower bounds,
    not precise peak predictions. The helper should choose sane fallbacks near
    the measured boundary without inventing half-terabyte numbers.
    """

    model_b = max(model_size_billions(config.model_id), 0.1)
    quant_factor = 4.0 / float(config.quant_bits)
    sequence_len = effective_sequence_length(config)
    if _is_qwen35_9b_hybrid(config.model_id):
        return _estimate_qwen35_9b_peak_gb(config, sequence_len, quant_factor)

    trajectory_factor = 1.0
    if config.max_turns > 1:
        single_turn_length = config.max_prompt_len + config.max_completion_len
        trajectory_factor = (sequence_len / max(single_turn_length, 1)) ** 0.35
    peak = max(
        0.75 + 0.55 * model_b * quant_factor,
        6.2
        * (model_b / 0.6) ** 0.72
        * quant_factor
        * (_memory_group_size(config) / 4.0)
        * (config.max_completion_len / 256.0) ** 0.75
        * trajectory_factor,
    )
    return float(peak)


def memory_confidence(config: TrainConfig) -> str:
    """Describe whether the memory estimate is anchored or extrapolated."""

    if _near_measured_anchor(config):
        return "measured-anchor"
    if config.max_turns > 1:
        return "estimated — multi-turn OOM risk, not measured"
    if _is_qwen35_9b_hybrid(config.model_id) and not config.gradient_checkpointing:
        return "estimated — OOM risk, not measured"
    return "estimated"


def memory_reduction_hint(config: TrainConfig) -> str | None:
    """Return the first knob to reduce when memory is likely too high."""

    if _is_qwen35_9b_hybrid(config.model_id) and not config.gradient_checkpointing:
        return "enable gradient_checkpointing"
    if config.rollout_mode == "parallel_per_turn" and config.group_size > 1:
        return "set rollout_mode='sequential' or reduce group_size"
    if config.group_size > 1:
        return "reduce group_size"
    if config.max_completion_len > 32:
        return "reduce max_completion_len"
    return None


def _near_measured_anchor(config: TrainConfig) -> bool:
    sequence_len = effective_sequence_length(config)
    if (
        config.model_id == "mlx-community/Qwen3-0.6B-4bit"
        and config.group_size == 4
        and config.max_turns == 1
        and 200 <= config.max_completion_len <= 320
    ):
        return True
    if _is_qwen35_9b_hybrid(config.model_id):
        if config.gradient_checkpointing and config.group_size in {2, 4}:
            return 560 <= sequence_len <= 660
        return (
            not config.gradient_checkpointing
            and config.group_size == 2
            and 96 <= sequence_len <= 160
        )
    return False


def _is_qwen35_9b_hybrid(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen3.5" in normalized and "9b" in normalized


def _estimate_qwen35_9b_peak_gb(
    config: TrainConfig,
    sequence_len: int,
    quant_factor: float,
) -> float:
    sequence_factor = max(sequence_len, 1) / 609.0
    group_size = _memory_group_size(config)
    if config.gradient_checkpointing:
        resident_gb = 5.9 * quant_factor
        per_group_gb = 10.0 * sequence_factor * quant_factor
        return resident_gb + per_group_gb * group_size

    if sequence_len <= 160 and group_size <= 2:
        return (5.6 + 0.2375 * sequence_len) * quant_factor

    checkpointed_same_shape = _estimate_qwen35_9b_peak_gb(
        replace(config, gradient_checkpointing=True),
        sequence_len,
        quant_factor,
    )
    oom_floor = 72.0 * quant_factor
    return max(oom_floor, checkpointed_same_shape * 1.35)


def _memory_group_size(config: TrainConfig) -> int:
    if config.max_turns > 1 and config.rollout_mode == "sequential":
        return max(1, min(config.group_size, config.micro_batch_size or 1))
    return config.group_size


def memory_fit(config: TrainConfig, available_unified_memory_gb: float) -> MemoryEstimate:
    """Estimate whether a config fits and suggest the largest simple fallback."""

    config.validate()
    if available_unified_memory_gb <= 0:
        raise ConfigError("available_unified_memory_gb must be positive.")
    estimate = estimate_peak_memory_gb(config)
    confidence = memory_confidence(config)
    reduction_hint = memory_reduction_hint(config)
    if estimate <= available_unified_memory_gb:
        return MemoryEstimate(
            estimated_peak_gb=estimate,
            available_gb=available_unified_memory_gb,
            fits=True,
            suggested_config=config,
            suggested_peak_gb=estimate,
            suggestions=(),
            confidence=confidence,
            reduction_hint=reduction_hint,
        )

    candidates: list[tuple[float, int, int, bool, TrainConfig]] = []
    for checkpoint in {config.gradient_checkpointing, True}:
        for group_size in range(1, config.group_size + 1):
            for tokens in range(32, config.max_completion_len + 1, 32):
                candidate = replace(
                    config,
                    group_size=group_size,
                    max_completion_len=tokens,
                    gradient_checkpointing=checkpoint,
                )
                candidate_peak = estimate_peak_memory_gb(candidate)
                if candidate_peak <= available_unified_memory_gb:
                    candidates.append(
                        (
                            group_size * tokens,
                            group_size,
                            tokens,
                            checkpoint,
                            candidate,
                        )
                    )
    if not candidates:
        candidate = replace(
            config,
            group_size=1,
            max_completion_len=32,
            gradient_checkpointing=True,
        )
    else:
        candidate = max(candidates, key=lambda item: item[:4])[4]

    suggestions: list[str] = []
    if candidate.group_size != config.group_size:
        suggestions.append(f"set group_size={candidate.group_size}")
    if candidate.gradient_checkpointing != config.gradient_checkpointing:
        suggestions.append("enable gradient_checkpointing")
    if candidate.max_completion_len != config.max_completion_len:
        suggestions.append(f"set max_completion_len={candidate.max_completion_len}")
    suggested_peak = estimate_peak_memory_gb(candidate)
    return MemoryEstimate(
        estimated_peak_gb=estimate,
        available_gb=available_unified_memory_gb,
        fits=False,
        suggested_config=candidate,
        suggested_peak_gb=suggested_peak,
        suggestions=tuple(suggestions),
        confidence=confidence,
        reduction_hint=reduction_hint,
    )
