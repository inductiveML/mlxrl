from __future__ import annotations

import subprocess
import sys

import pytest

from mlxrl.config import ConfigError, validate_output_path


@pytest.mark.parametrize("tag", ["answer", "action"])
@pytest.mark.parametrize("kind", ["whitespace", "repeated_opening"])
def test_malformed_model_output_finishes_within_a_bounded_process(tag: str, kind: str) -> None:
    # A separate process bounds failures: the old regex hangs on these inputs.
    script = """
import sys
from mlxrl.data.rewards import format_reward
from mlxrl.env import default_action_parser
tag, kind = sys.argv[1:]
text = '<' + tag + '>' + ' ' * 100_000 + 'X'
if kind == 'repeated_opening':
    text = ('<' + tag + '>') * 20_000 + 'X'
if tag == 'answer':
    assert format_reward(text) == 0.0
else:
    assert default_action_parser(text) == text.strip()
"""
    subprocess.run([sys.executable, "-c", script, tag, kind], check=True, timeout=5)


@pytest.mark.parametrize("existing_file", [False, True])
def test_output_path_rejects_symlinks_outside_the_working_directory(
    tmp_path, monkeypatch, existing_file: bool
) -> None:
    working = tmp_path / "working"
    working.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if existing_file:
        (outside / "output.npz").touch()
        (working / "output.npz").symlink_to(outside / "output.npz")
        relative = "output.npz"
    else:
        (working / "link").symlink_to(outside, target_is_directory=True)
        relative = "link/output.npz"
    monkeypatch.chdir(working)

    with pytest.raises(ConfigError, match="inside the working directory"):
        validate_output_path(relative)


def test_output_path_accepts_symlinks_that_remain_inside_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "outputs", target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    assert str(validate_output_path("link/model.npz")) == "link/model.npz"


def test_output_path_rejects_symlink_cycles(tmp_path, monkeypatch) -> None:
    (tmp_path / "cycle").symlink_to("cycle")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="resolved safely"):
        validate_output_path("cycle/model.npz")
