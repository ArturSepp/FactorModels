"""Installed identity contracts for the clean FactorModels namespace migration."""
import ast
from importlib import metadata, util
from pathlib import Path

import factormodels
import factormodels.analytics as analytics
import factormodels.estimation as estimation


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
    for name in ("constraints", "construction", "examples", "model", "risk", "signals", "portfolio_context", "spec", "premia", "cma"):
        assert not (package / name).exists(), name
    assert not hasattr(factormodels, "synthetic_prices")
    assert "synthetic_prices" not in factormodels.__all__


def test_stage2_owners_and_retired_paths():
    """Construction/contracts are analytics; FL-backed scoring is estimation."""
    for name, module in {
        "FactorModelSpec": "definitions", "build_factor_panel": "construction",
        "compute_beta_loading_signs": "beta_constraints", "FactorModelSnapshot": "models",
        "RollingFactorModel": "models", "build_risk_model": "risk",
        "prepare_portfolio_context": "portfolio_context",
    }.items():
        assert getattr(analytics, name) is getattr(factormodels, name)
        assert getattr(analytics, name).__module__ == f"factormodels.analytics.{module}"
        assert name in analytics.__all__
    for name in ("extract_rolling_clusters", "prepare_cluster_scores"):
        assert getattr(estimation, name) is getattr(factormodels, name)
        assert getattr(estimation, name).__module__ == "factormodels.estimation.cluster_scores"
        assert name in estimation.__all__
    for retired in ("spec", "estimation.models", "estimation.risk", "estimation.beta_constraints", "analytics.signals"):
        assert util.find_spec(f"factormodels.{retired}") is None


def test_gls_consolidation_has_one_owner_and_canonical_shipped_tests():
    """Wheels expose final owners without old namespaces or accounting delegates."""
    from factormodels import gls_diagnostic as gls
    from factormodels.gls_diagnostic import sharpe_accounting as sa
    assert gls.compute_gls_geometry.__module__ == "factormodels.gls_diagnostic.geometry"
    assert gls.infer_factor_premia.__module__ == "factormodels.gls_diagnostic.inference"
    for name in gls.__all__:
        assert getattr(gls, name) is getattr(factormodels, name)
    for name in ("_gls_geometry", "_loadings", "gls_projector", "gls_split", "factor_mimicking_portfolios"):
        assert not hasattr(sa, name)
    for old in ("premia", "cma", "analytics.opportunity", "analytics.uncertainty", "analytics.universe_diagnostics"):
        assert util.find_spec("factormodels." + old) is None
    package = Path(factormodels.__file__).parent
    assert not list(package.rglob("*_test.py"))
    for owner in ("analytics", "estimation", "gls_diagnostic", "reports"):
        assert list((package / owner / "tests").glob("test_*.py")), owner
