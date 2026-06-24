from __future__ import annotations

import ast
from pathlib import Path


def test_engine_layers_do_not_import_algorithm_implementations() -> None:
    root = Path(__file__).resolve().parents[1] / "mlxrl"
    checked_roots = [root / "rollout", root / "policy", root / "train"]
    violations: list[str] = []

    for checked_root in checked_roots:
        for path in checked_root.rglob("*.py"):
            module = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    if imported == "mlxrl.algo" or imported.startswith("mlxrl.algo."):
                        violations.append(f"{path.relative_to(root)} imports {imported}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        if imported == "mlxrl.algo" or imported.startswith("mlxrl.algo."):
                            violations.append(f"{path.relative_to(root)} imports {imported}")

    assert violations == []

