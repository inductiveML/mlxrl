from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.llama import Model, ModelArgs

from mlxrl import cli
from mlxrl.algo.grpo import GRPOAlgorithm
from mlxrl.config import TrainConfig
from mlxrl.data.gsm8k import MINI_GSM8K
from mlxrl.data.rewards import extract_answer
from mlxrl.policy.model import LoRAConfig, inject_lora_adapters
from mlxrl.rollout.naive import Completion

pytestmark = pytest.mark.metal
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/qwen3_0_6b_confidence.toml"


def test_confidence_example_keeps_prompt_and_algorithm_fixed_when_disabled():
    enabled = TrainConfig.from_file(EXAMPLE)
    disabled = replace(enabled, confidence=replace(enabled.confidence, enabled=False))
    disabled.validate()
    assert cli._algorithm_from_config(enabled) == cli._algorithm_from_config(disabled)
    algorithm = cli._algorithm_from_config(enabled)
    assert isinstance(algorithm, GRPOAlgorithm) and algorithm.sequence_mean_reduction
    assert enabled.groups_per_step == 4
    for i in range(len(MINI_GSM8K)):
        a = cli._gsm8k_prompt(cli._namespace_from_train_config(enabled), i)
        b = cli._gsm8k_prompt(cli._namespace_from_train_config(disabled), i)
        assert a == b
        assert "step: expression = numeric_result" in a[0]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_cli_multi_prompt_credit_handoff_and_artifact(monkeypatch, tmp_path, checkpoint):
    """Controlled completions, real 4-bit forwards/Adam, and the actual CLI loop."""
    mx.random.seed(24)
    model = Model(ModelArgs(
        model_type="llama", hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-5, vocab_size=128, max_position_embeddings=256,
    ))
    nn.quantize(model, group_size=32, bits=4)
    inject_lora_adapters(model, LoRAConfig(rank=4, scale=2, dropout=0))
    mx.eval(model.parameters())
    model.eval()

    class Tokenizer:
        pad_token_id = 0

        def decode(self, tokens):
            return "".join(chr(x) for x in tokens)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_policy_with_lora", lambda **kw: (model, Tokenizer(), None))
    config = replace(
        TrainConfig.from_file(EXAMPLE), group_size=2, groups_per_step=2,
        steps=9, gradient_checkpointing=checkpoint, output="conf.npz",
    )
    config.validate()
    args = cli._namespace_from_train_config(config)
    args._algorithm_override = cli._algorithm_from_config(config)
    seen = []

    def rollouts(**kwargs):
        assert len(kwargs["prompts"]) == 2
        rows = []
        for i, prompt in enumerate(kwargs["prompts"]):
            example = next(e for e in MINI_GSM8K if e.question in prompt)
            correct = extract_answer(example.answer)
            for j, answer in enumerate((correct, "-999")):
                text = f"step: 2 + 3 = 5\nstep: 5 + 4 = 9\n<answer>{answer}</answer>"
                rows.append(Completion(i, j, (1, 2), tuple(map(ord, text)), (), text))
        return rows

    original_step = cli.optimizer_step

    def optimizer_step(**kwargs):
        update = len(seen) + 1
        batch, optimizer = kwargs["batch"], kwargs["optimizer"]
        np.testing.assert_allclose(batch.rewards, [1.1, 0.1, 1.1, 0.1])
        assert (batch.confidence_logprobs is not None) == (update <= 8)
        assert (batch.token_advantages is not None) == (update <= 8)
        assert int(optimizer.step) == update - 1
        seen.append(id(optimizer))
        metrics = original_step(**kwargs)
        assert int(optimizer.step) == update
        return metrics

    monkeypatch.setattr(cli, "generate_group_rollouts", rollouts)
    monkeypatch.setattr(cli, "optimizer_step", optimizer_step)
    assert cli._phase1_gsm8k(args) == 0
    assert len(seen) == 9 and len(set(seen)) == 1
    with np.load(tmp_path / "conf.npz") as artifact:
        np.testing.assert_array_equal(artifact["conf_updates"], np.arange(1, 10))
        assert np.all(artifact["conf_lambda"][:8] > 0)
        assert artifact["conf_lambda"][8] == 0
        assert artifact["conf_previous_lambda"] > 0
        assert "token_advantages" not in artifact.files
        assert all(np.all(np.isfinite(artifact[key])) for key in artifact.files)
