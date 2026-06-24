from __future__ import annotations

import argparse

from mlxrl.algo.gigpo import GiGPOAlgorithm
from mlxrl.algo.grpo import DAPOAlgorithm, DrGRPOAlgorithm, GSPOAlgorithm
from mlxrl.cli import (
    _algorithm_from_args,
    _algorithm_from_config,
    _env_factory_from_config,
    _trajectory_algorithm_from_config,
)
from mlxrl.config import AlgorithmConfig, TrainConfig


def test_algorithm_from_args_and_config_share_dapo_defaults() -> None:
    args = argparse.Namespace(
        algorithm="dapo",
        clip_low="default",
        clip_high="default",
        dapo_dynamic_sampling=True,
        gspo_importance="sequence",
        drgrpo_normalize_rewards=False,
        drgrpo_loss_reduction="sequence_max_tokens",
        drgrpo_max_tokens=None,
        max_tokens=128,
    )
    config = TrainConfig(
        max_completion_len=128,
        algorithm=AlgorithmConfig(name="dapo", dapo_dynamic_sampling=True),
    )

    from_args = _algorithm_from_args(args)
    from_config = _algorithm_from_config(config)

    assert isinstance(from_args, DAPOAlgorithm)
    assert isinstance(from_config, DAPOAlgorithm)
    assert from_args.clip_low == from_config.clip_low == 0.2
    assert from_args.clip_high == from_config.clip_high == 0.28
    assert from_args.dynamic_sampling is from_config.dynamic_sampling is True


def test_algorithm_from_args_and_config_share_drgrpo_max_token_default() -> None:
    args = argparse.Namespace(
        algorithm="dr-grpo",
        clip_low="default",
        clip_high="default",
        dapo_dynamic_sampling=False,
        gspo_importance="sequence",
        drgrpo_normalize_rewards=True,
        drgrpo_loss_reduction="token_mean",
        drgrpo_max_tokens=None,
        max_tokens=96,
    )
    config = TrainConfig(
        max_completion_len=96,
        algorithm=AlgorithmConfig(
            name="dr-grpo",
            drgrpo_normalize_rewards=True,
            drgrpo_loss_reduction="token_mean",
        ),
    )

    from_args = _algorithm_from_args(args)
    from_config = _algorithm_from_config(config)

    assert isinstance(from_args, DrGRPOAlgorithm)
    assert isinstance(from_config, DrGRPOAlgorithm)
    assert from_args.normalize_rewards is from_config.normalize_rewards is True
    assert from_args.loss_reduction == from_config.loss_reduction == "token_mean"
    assert from_args.max_tokens == from_config.max_tokens == 96


def test_algorithm_from_args_and_config_share_gspo_defaults() -> None:
    args = argparse.Namespace(
        algorithm="gspo",
        clip_low="default",
        clip_high="default",
        dapo_dynamic_sampling=False,
        gspo_importance="token",
        drgrpo_normalize_rewards=False,
        drgrpo_loss_reduction="sequence_max_tokens",
        drgrpo_max_tokens=None,
        max_tokens=64,
    )
    config = TrainConfig(
        algorithm=AlgorithmConfig(name="gspo", gspo_importance="token"),
    )

    from_args = _algorithm_from_args(args)
    from_config = _algorithm_from_config(config)

    assert isinstance(from_args, GSPOAlgorithm)
    assert isinstance(from_config, GSPOAlgorithm)
    assert from_args.importance == from_config.importance == "token"
    assert from_args.clip_low == from_config.clip_low == 3e-4
    assert from_args.clip_high == from_config.clip_high == 4e-4


def test_algorithm_from_args_preserves_disabled_clipping() -> None:
    args = argparse.Namespace(
        algorithm="dapo",
        clip_low="none",
        clip_high="none",
        dapo_dynamic_sampling=False,
        gspo_importance="sequence",
        drgrpo_normalize_rewards=False,
        drgrpo_loss_reduction="sequence_max_tokens",
        drgrpo_max_tokens=None,
        max_tokens=64,
    )

    algorithm = _algorithm_from_args(args)

    assert isinstance(algorithm, DAPOAlgorithm)
    assert algorithm.clip_low is None
    assert algorithm.clip_high is None


def test_trajectory_algorithm_from_config_builds_gigpo() -> None:
    config = TrainConfig(
        algorithm=AlgorithmConfig(
            name="gigpo",
            clip_low=0.1,
            clip_high=0.2,
            gigpo_omega=0.5,
            gigpo_gamma=0.9,
            gigpo_normalization="center",
        )
    )

    algorithm = _trajectory_algorithm_from_config(config)

    assert isinstance(algorithm, GiGPOAlgorithm)
    assert algorithm.omega == 0.5
    assert algorithm.gamma == 0.9
    assert algorithm.normalization == "center"
    assert algorithm.clip_low == 0.1
    assert algorithm.clip_high == 0.2


def test_env_factory_from_config_builds_reference_env() -> None:
    config = TrainConfig(env_name="recurring-text", max_turns=2)
    env = _env_factory_from_config(config)("task", 0, 0)

    assert env.max_turns == 2
    assert env.reset() == "task=task; state=start"
