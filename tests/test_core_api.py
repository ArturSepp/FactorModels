"""Public documentation, dependency direction and completed API contracts."""

import ast
import json
from pathlib import Path

import factormodels as fm


def test_core_api_is_documented_and_resolves():
    """Every canonical export has an owning documented entry and resolves."""
    root = Path(__file__).resolve().parents[1]
    documentation = (root / "docs" / "api.md").read_text(encoding="utf-8")
    assert len(fm.__all__) == len(set(fm.__all__))
    for name in fm.__all__:
        obj = getattr(fm, name)
        assert obj.__doc__, name
        assert f"`{name}`" in documentation, name
    assert fm.FactorCovarEstimator.__module__ == "factormodels.estimation.estimator"


def test_library_has_no_consumer_dependency():
    """Installed numerical modules never import an optimiser or private consumer."""
    package = Path(fm.__file__).resolve().parent
    forbidden = {"optimalportfolios", "ramen", "privateassets"}
    for path in package.rglob("*.py"):
        if "tests" in path.relative_to(package).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not (set(m.split(".")[0] for m in modules) & forbidden), str(path)


def test_frozen_root_api_survives():
    """Retain the reviewed 0.2.0 root exports alongside subsequent additions."""
    frozen = json.loads((Path(__file__).with_name("root_api_0_2_0.json")).read_text())
    assert frozen["version"] == "0.2.0"
    assert len(frozen["names"]) == len(set(frozen["names"]))
    assert set(frozen["names"]) <= set(fm.__all__)
    assert all(hasattr(fm, name) for name in frozen["names"])
