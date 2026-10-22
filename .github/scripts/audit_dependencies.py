"""Audit every registry version in uv.lock, including other-platform dependencies."""

from __future__ import annotations

import subprocess
import tempfile
import tomllib
from pathlib import Path


def main() -> None:
    lock = tomllib.loads(Path("uv.lock").read_text())
    groups: list[dict[str, str]] = []
    for package in lock["package"]:
        if package["source"] == {"editable": "."}:
            continue
        if package["source"].get("registry") != "https://pypi.org/simple":
            raise ValueError(f"Unrecognized dependency source for {package['name']}.")
        hashes = {wheel["hash"] for wheel in package.get("wheels", [])}
        if "sdist" in package:
            hashes.add(package["sdist"]["hash"])
        if not hashes:
            raise ValueError(f"Missing locked hashes for {package['name']}.")
        requirement = f"{package['name']}=={package['version']} " + " ".join(
            f"--hash={digest}" for digest in sorted(hashes)
        )
        # pip-audit rejects duplicate names even when versions target other Pythons.
        for group in groups:
            if package["name"] not in group:
                group[package["name"]] = requirement
                break
        else:
            groups.append({package["name"]: requirement})
    with tempfile.TemporaryDirectory(prefix="mlxrl-dependency-audit-") as directory:
        for index, group in enumerate(groups):
            requirements = Path(directory) / f"requirements-{index}.txt"
            requirements.write_text("\n".join(group.values()) + "\n")
            subprocess.run(
                [
                    "pip-audit",
                    "--strict",
                    "--disable-pip",
                    "--require-hashes",
                    "-r",
                    str(requirements),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
