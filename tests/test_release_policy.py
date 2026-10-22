from __future__ import annotations

import io
import json
import runpy
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/verify_release.py"
verify_release = runpy.run_path(str(SCRIPT))["verify_release"]


def git(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Security Test",
            "-c",
            "user.email=security@example.invalid",
            *args,
        ],
        text=True,
    ).strip()


@pytest.fixture
def release_repo(tmp_path, monkeypatch) -> tuple[str, dict[str, Any]]:
    monkeypatch.chdir(tmp_path)
    git("init", "-q", "-b", "main")
    git("commit", "-q", "--allow-empty", "-m", "reviewed main commit")
    sha = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", sha)
    git("tag", "-a", "v0.2.4", "-m", "release")
    run = {
        "head_sha": sha,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/ci.yml",
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps({"workflow_runs": [run]}).encode()),
    )
    return sha, run


def test_release_accepts_an_annotated_tag_from_main_after_ci(release_repo) -> None:
    sha, _ = release_repo
    verify_release("refs/tags/v0.2.4", sha, "0.2.4", "owner/repo", "test-token")


def test_release_rejects_a_version_mismatch(release_repo) -> None:
    sha, _ = release_repo
    with pytest.raises(ValueError, match="match the package version"):
        verify_release("refs/tags/v0.2.5", sha, "0.2.4", "owner/repo", "test-token")


def test_release_rejects_a_tagged_commit_outside_main(release_repo) -> None:
    git("commit", "-q", "--allow-empty", "-m", "unreviewed commit")
    sha = git("rev-parse", "HEAD")
    git("tag", "-f", "-a", "v0.2.4", "-m", "unreviewed tag")
    with pytest.raises(ValueError, match="already be on origin/main"):
        verify_release("refs/tags/v0.2.4", sha, "0.2.4", "owner/repo", "test-token")


def test_release_rejects_a_workflow_commit_different_from_the_tag(release_repo) -> None:
    with pytest.raises(ValueError, match="workflow commit differ"):
        verify_release("refs/tags/v0.2.4", "a" * 40, "0.2.4", "owner/repo", "test-token")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_sha", "a" * 40),
        ("head_branch", "feature"),
        ("event", "pull_request"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_release_requires_successful_main_push_ci(release_repo, field: str, value: str) -> None:
    sha, run = release_repo
    run[field] = value
    with pytest.raises(ValueError, match="successful push CI on main"):
        verify_release("refs/tags/v0.2.4", sha, "0.2.4", "owner/repo", "test-token")


def test_branch_dispatch_can_only_build() -> None:
    verify_release("refs/heads/feature", "a" * 40, "0.2.4", "owner/repo", "")
