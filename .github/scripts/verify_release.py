"""Require versioned tags to identify a main commit with successful push CI."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path


def verify_release(ref: str, sha: str, version: str, repository: str, token: str) -> None:
    if not ref.startswith("refs/tags/"):
        return
    if not re.fullmatch(r"refs/tags/v[0-9]+\.[0-9]+\.[0-9]+", ref):
        raise ValueError("Release tags must use vX.Y.Z.")
    if ref != f"refs/tags/v{version}":
        raise ValueError("Release tag must match the package version.")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Release commit must be a full SHA.")
    tagged = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    if tagged != sha:
        raise ValueError("Release tag and workflow commit differ.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"]).returncode:
        raise ValueError("Release commit must already be on origin/main.")
    query = urllib.parse.urlencode(
        {"head_sha": sha, "branch": "main", "event": "push", "status": "success"}
    )
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/workflows/ci.yml/runs?{query}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        runs = json.load(response)["workflow_runs"]
    if not any(
        run["head_sha"] == sha
        and run["head_branch"] == "main"
        and run["event"] == "push"
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and run["path"] == ".github/workflows/ci.yml"
        for run in runs
    ):
        raise ValueError("Release commit needs successful push CI on main.")


def main() -> None:
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    module = ast.parse(Path("mlxrl/__init__.py").read_text())
    package_version = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    )
    if package_version != version:
        raise ValueError("Package and distribution versions must match.")
    verify_release(
        os.environ["GITHUB_REF"],
        os.environ["GITHUB_SHA"],
        version,
        os.environ["GITHUB_REPOSITORY"],
        os.environ.get("GH_TOKEN", ""),
    )


if __name__ == "__main__":
    main()
