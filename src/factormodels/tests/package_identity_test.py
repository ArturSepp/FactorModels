"""Installed identity contracts for the clean FactorModels namespace migration."""
import ast
from importlib import metadata, util
from pathlib import Path

import factormodels


def test_canonical_distribution_has_no_legacy_forwarding_package():
    """A clean installation exposes FactorModels without the removed namespace."""
    assert metadata.distribution("factormodels").metadata["Name"] == "factormodels"
    assert util.find_spec("tradablefactors") is None
    assert factormodels.FactorModelSpec.__name__ == "FactorModelSpec"
    assert factormodels.FactorCovarEstimator.__name__ == "FactorCovarEstimator"
    assert factormodels.FactorModelSnapshot.__name__ == "FactorModelSnapshot"
    assert factormodels.RollingFactorModel.__name__ == "RollingFactorModel"


def test_removed_input_bundle_has_no_module_or_public_exports():
    """A source or wheel installation exposes only direct estimator input APIs."""
    assert util.find_spec("factormodels.bundle") is None
    for name in ("BUNDLE_SCHEMA_VERSION", "BundleSettings", "FactorBundle", "load_bundle",
                 "fit_bundle", "save_bundle", "synthetic_bundle"):
        assert not hasattr(factormodels, name)
        assert name not in factormodels.__all__


def test_initializers_export_named_modules_without_implementation():
    """Source and wheels keep implementations out of init files and redundant packages."""
    package = Path(factormodels.__file__).parent
    assert sorted(path.name for path in package.glob("*.py")) == ["__init__.py"]
    for path in package.rglob("__init__.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            allowed = isinstance(node, (ast.Import, ast.ImportFrom))
            allowed |= (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
            allowed |= (isinstance(node, ast.Assign)
                        and all(isinstance(target, ast.Name) and target.id == "__all__"
                                for target in node.targets))
            assert allowed, (path, node.lineno, type(node).__name__)
        if path.parent != package:
            assert any(item != path for item in path.parent.rglob("*.py")), path
    for name in ("constraints", "construction", "examples", "model", "risk", "signals", "portfolio_context"):
        assert not (package / name).exists(), name
    assert not hasattr(factormodels, "synthetic_prices")
    assert "synthetic_prices" not in factormodels.__all__
