from __future__ import annotations

import json
from argparse import Namespace

from benchmarks.audit_phase4_config import (
    FastPathInspection,
    build_audit_rows,
    render_markdown,
)
from benchmarks.run_phase4 import (
    BenchResult,
    enforce_gradient_timing_sanity,
    format_result_line,
    render_summary,
    result_from_json,
)


def test_result_from_json_backfills_unified_token_denominator() -> None:
    result = result_from_json(
        json.dumps(
            {
                "target": "legacy",
                "pass_index": 0,
                "status": "ok",
                "model": "toy",
                "steps": 1,
                "warmup_steps": 0,
                "rollout_tokens": 12,
                "rollout_seconds": 3.0,
            }
        ),
        target="legacy",
        pass_index=0,
    )

    assert result.generated_completion_tokens == 12
    assert result.prompt_tokens == 0
    assert result.total_forward_tokens == 12
    assert result.tok_s_denominator == 12
    assert result.rollout_tokens == 12
    assert result.rollout_tok_s == 4.0


def test_summary_and_result_line_show_unified_token_fields() -> None:
    result = BenchResult(
        target="mlxrl",
        pass_index=0,
        status="ok",
        model="toy",
        steps=1,
        warmup_steps=0,
        generated_completion_tokens=8,
        prompt_tokens=3,
        total_forward_tokens=11,
        tok_s_denominator=8,
        rollout_tokens=8,
        rollout_seconds=2.0,
        rollout_tok_s=4.0,
    )

    summary = render_summary([result])
    line = format_result_line(result)

    assert "gen toks" in summary
    assert "prompt toks" in summary
    assert "tok/s denom" in summary
    assert "Rollout tok/s definition" in summary
    assert "Sync guarantee" in summary
    assert "apples-to-apples" in summary
    assert "tok_s_denom=8" in line
    assert "gen_tokens=8" in line
    assert "prompt_tokens=3" in line


def test_sub_10ms_gradient_step_is_rejected() -> None:
    result = BenchResult(
        target="mlx-tune",
        pass_index=0,
        status="ok",
        model="toy",
        steps=1,
        warmup_steps=0,
        gradient_steps=1,
        gradient_seconds=0.005,
        gradient_step_s=0.005,
    )

    checked = enforce_gradient_timing_sanity(result)

    assert checked.status == "missing"
    assert checked.gradient_seconds == 0.0
    assert checked.gradient_step_s is None
    assert "rejected synced grad_s_step" in checked.note


def test_gate4_audit_flags_mlx_tune_sampler_caveat() -> None:
    args = Namespace(
        model="mlx-community/Qwen3-0.6B-4bit",
        batch_size=1,
        group_size=4,
        max_context=4096,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        min_p=0.0,
        beta=0.04,
        wired_limit_mb=32768,
    )
    inspection = FastPathInspection(
        mlx_tune_prefix_cache=True,
        mlx_tune_stream_generate=True,
        mlx_tune_generation_uses_top_p=False,
        mlx_tune_generation_uses_top_k=False,
        mlx_tune_generation_uses_min_p=False,
        mlx_lm_lora_batch_generate=True,
        mlx_lm_lora_sampler_knobs=True,
    )

    rows = build_audit_rows(args, ("mlx-tune", "mlx-lm-lora"), inspection)
    markdown = render_markdown(rows)
    by_target = {row.target: row for row in rows}

    assert by_target["mlx-tune"].status == "package-speed only"
    assert "top_p/top_k/min_p are not used" in by_target["mlx-tune"].caveat
    assert by_target["mlx-lm-lora"].top_k == "0"
    assert "mlx-lm-lora" in markdown
