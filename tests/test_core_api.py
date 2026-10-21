"""Public documentation, dependency direction and completed API contracts."""

import ast
from pathlib import Path

import factormodels as tf


def test_core_api_is_documented_and_resolves():
    """Every canonical export has an owning documented entry and resolves."""
    root = Path(__file__).resolve().parents[1]
    documentation = (root / "docs" / "api.md").read_text(encoding="utf-8")
    assert len(tf.__all__) == len(set(tf.__all__))
    for name in tf.__all__:
        obj = getattr(tf, name)
        assert obj.__doc__, name
        assert f"`{name}`" in documentation, name
    assert tf.FactorCovarEstimator.__module__ == "factormodels.estimation"


def test_library_has_no_consumer_dependency():
    """Installed numerical modules never import an optimiser or private consumer."""
    package = Path(tf.__file__).resolve().parent
    forbidden = {"optimalportfolios", "ramen", "privateassets"}
    for path in package.rglob("*.py"):
        if "tests" in path.relative_to(package).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not (set(m.split(".")[0] for m in modules) & forbidden), str(path)
