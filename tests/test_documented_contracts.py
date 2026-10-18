"""Check documented completion, dates, native units and downstream independence."""
from pathlib import Path
from dataclasses import replace
import json
import runpy

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_completed_model_can_be_inspected_without_rerunning_producer(monkeypatch):
    """Result tables, authoritative covariance and QIS views remain usable after fit."""
    from tradablefactors import FactorCovarEstimator
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    result = namespace["run_local_test"]()
    snapshot = result["snapshot"]
    def forbidden(*args, **kwargs):
        raise AssertionError("completed output inspection must not refit")
    monkeypatch.setattr(FactorCovarEstimator, "fit_current", forbidden)
    covariance = snapshot.asset_covariance
    risk = snapshot.to_risk_model()
    pd.testing.assert_frame_equal(risk.covar[snapshot.cutoff], covariance)
    assert covariance.shape == (4, 4)
    assert snapshot.data.y_betas.shape == (4, 2)
    assert snapshot.provenance["asset_return_type"] == "log"
    assert snapshot.provenance["reference_ccy"] == "USD"
    assert snapshot.cutoff == pd.Timestamp("2024-01-31")
    np.testing.assert_allclose(np.diag(covariance),
                               np.diag(snapshot.data.y_betas @ snapshot.data.x_covar
                                       @ snapshot.data.y_betas.T + snapshot.residual_covariance))


def test_proxy_definition_has_exact_single_component_nav_returns():
    """A one-component long-only recipe preserves supplied monthly log moves."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["public_proxy_bundle"]()
    inputs = namespace["synthetic_bundle"]().factor_levels.reindex(bundle.factor_levels.index)
    expected = np.log(inputs).diff().dropna()
    actual = np.log(bundle.factor_levels).diff().dropna()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["native", "empirical", "zero_residual"])
def test_valid_model_completes_when_illustrative_report_is_unavailable(kind, tmp_path):
    """Do not relabel native premia or misstate full residual risk as diagonal GLS."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    if kind == "native":
        settings = replace(bundle.settings, factor_return_types={"Equity": "LOG", "Rates": "DIFFERENCE"},
                           factor_units={"Equity": "dimensionless", "Rates": "index_points"})
    elif kind == "empirical":
        settings = replace(bundle.settings, residual_type="empirical", residual_covar_freq="ME")
    else:
        settings = replace(bundle.settings, residual_var_weight=0.)
    result = namespace["inspect_fit"](replace(bundle, settings=settings))
    assert result["snapshot"].asset_covariance.shape == (4, 4)
    assert result["audit"] is None and result["report_status"].startswith("unavailable:")
    output = namespace["persist_example"](result, tmp_path / kind)
    assert not (output / "cma_report.pdf").exists()
    metadata = json.loads((output / "result_metadata.json").read_text(encoding="utf-8"))
    assert metadata["report_status"] == result["report_status"]
    assert metadata["factor_units"] == dict(settings.factor_units)


def test_illustrative_audit_uses_selected_residual_variance_weight():
    """A nondefault residual weight reconciles the supplied production covariance."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    bundle = replace(bundle, settings=replace(bundle.settings, residual_var_weight=.5))
    result = namespace["inspect_fit"](bundle)
    validation = result["audit"].validation
    assert "fail" not in validation["status"].values
