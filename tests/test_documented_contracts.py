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
    from factormodels import FactorCovarEstimator
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


def test_log_fit_connects_to_explicit_arithmetic_forecasts_without_refitting(monkeypatch):
    """A frozen log-risk model supports the stated arithmetic mean-risk convention."""
    from factormodels import FactorCovarEstimator, fit_bundle
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    assert (bundle.domestic_rates == 0.).all().all()
    log_returns = np.log(bundle.asset_prices).diff().dropna()
    simple_returns = bundle.asset_prices.pct_change(fill_method=None).dropna()
    assert not np.allclose(log_returns.cov(), simple_returns.cov(), rtol=1e-8, atol=1e-12)
    snapshot = fit_bundle(bundle)
    before = (snapshot.loadings, snapshot.factor_covariance, snapshot.residual_covariance,
              snapshot.asset_covariance, snapshot.data.residuals.copy())
    def forbidden(*args, **kwargs):
        raise AssertionError("the arithmetic forecast audit must not refit")
    monkeypatch.setattr(FactorCovarEstimator, "fit_current", forbidden)
    result = namespace["audit_snapshot"](bundle, snapshot)
    assert result["audit"] is not None and result["report_status"].startswith("available:")
    inputs, audit, forecasts = result["inputs"], result["audit"], result["forecasts"]
    for actual, expected in zip((inputs.betas, inputs.factor_covariance,
                                 snapshot.residual_covariance, inputs.production_covariance,
                                 snapshot.data.residuals), before):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_allclose(inputs.residual_variances, np.diag(before[2]), rtol=0., atol=0.)
    np.testing.assert_allclose(audit.assets.excess_cma, forecasts.provider_total-forecasts.cash)
    np.testing.assert_allclose(audit.factors.calibrated_premium, result["premia"].premia)
    np.testing.assert_allclose(inputs.betas @ inputs.factor_premia + inputs.channels.sum(axis=1),
                               inputs.excess_cma, atol=1e-12)
    assert audit.summary["accounting_status"] == "pass"
    assert audit.provenance["risk_basis"] == "annualized_log_return_covariance"
    assert "not an exact annual compounded-return distribution" in audit.provenance["mean_risk_convention"]
    future_rates = pd.DataFrame(.03, index=[snapshot.cutoff+pd.offsets.MonthEnd(1)],
                                columns=bundle.domestic_rates.columns)
    future_bundle = replace(bundle, domestic_rates=pd.concat([bundle.domestic_rates, future_rates]))
    future_result = namespace["audit_snapshot"](future_bundle, snapshot)
    pd.testing.assert_frame_equal(future_result["forecasts"], forecasts, check_exact=True)


def test_log_forecast_bridge_uses_full_variance_reconciliation_once():
    """Annual alpha and factor/asset variance corrections must not be rescaled twice."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    result = namespace["inspect_fit"](namespace["synthetic_bundle"]())
    snapshot, forecasts = result["snapshot"], result["forecasts"]
    b, f, sigma = snapshot.loadings, snapshot.factor_covariance, snapshot.asset_covariance
    # Independent geometric sum includes the zero initial state after leading missing data.
    residuals = snapshot.data.residuals
    assert residuals.iloc[0].isna().all()
    finite = residuals.dropna()
    assert residuals.loc[finite.index[0]:].notna().all().all()
    decay = 1.-2./37.
    weights = (1.-decay)*decay**np.arange(len(finite)-1, -1, -1)
    annual_alpha = finite.mul(weights, axis=0).sum()
    np.testing.assert_allclose(forecasts.historical_log_alpha, annual_alpha, rtol=1e-12, atol=1e-14)
    jensen = .5*(np.diag(sigma)-b @ np.diag(f))
    np.testing.assert_allclose(forecasts.credited_alpha, annual_alpha+jensen, atol=1e-12)
    assert not np.allclose(jensen, .5*np.diag(snapshot.residual_covariance), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(result["factor_assumptions"].arithmetic_premium, .04)
    np.testing.assert_allclose(forecasts.model_arithmetic_mean,
                               b @ result["factor_assumptions"].arithmetic_premium
                               + forecasts.credited_alpha, atol=1e-12)
    assert result["audit"].provenance["bridge_annualization_factor"] == 1.


@pytest.mark.parametrize("kind,reason", [
    ("native", "native difference factors"), ("empirical", "correlated residual risk"),
    ("zero_residual", "strictly positive residual variances"), ("nonzero_cash", "nonzero cash"),
])
def test_valid_model_completes_when_illustrative_report_is_unavailable(kind, reason, tmp_path):
    """Do not relabel native premia or misstate full residual risk as diagonal GLS."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    if kind == "native":
        settings = replace(bundle.settings, factor_return_types={"Equity": "LOG", "Rates": "DIFFERENCE"},
                           factor_units={"Equity": "dimensionless", "Rates": "index_points"})
    elif kind == "empirical":
        settings = replace(bundle.settings, residual_type="empirical", residual_covar_freq="ME")
    elif kind == "zero_residual":
        settings = replace(bundle.settings, residual_var_weight=0.)
    else:
        settings = bundle.settings
        bundle = replace(bundle, domestic_rates=bundle.domestic_rates+.03)
    result = namespace["inspect_fit"](replace(bundle, settings=settings))
    assert result["snapshot"].asset_covariance.shape == (4, 4)
    assert result["audit"] is None and result["report_status"].startswith("unavailable:")
    assert reason in result["report_status"]
    output = namespace["persist_example"](result, tmp_path / kind)
    assert not (output / "cma_report.pdf").exists()
    metadata = json.loads((output / "result_metadata.json").read_text(encoding="utf-8"))
    assert metadata["report_status"] == result["report_status"]
    assert metadata["factor_units"] == dict(settings.factor_units)
    residual = result["snapshot"].residual_covariance
    saved_residual = pd.read_csv(output / "residual_covariance.csv", index_col=0)
    pd.testing.assert_frame_equal(saved_residual, residual, atol=1e-15, rtol=1e-12)
    if kind == "empirical":
        assert np.max(np.abs(residual.to_numpy()-np.diag(np.diag(residual)))) > 1e-8
    elif kind == "zero_residual":
        np.testing.assert_array_equal(residual, 0.)


def test_arithmetic_audit_uses_selected_residual_variance_weight():
    """The same nondefault annual residual weight feeds GLS, covariance and audit."""
    namespace = runpy.run_path(str(ROOT / "examples" / "gls_cma_workflow.py"))
    result = namespace["arithmetic_example"](residual_var_weight=.5)
    inputs, audit = result["inputs"], result["audit"]
    expected_variances = .5*12*result["historical_residual"].var(ddof=0)
    np.testing.assert_allclose(inputs.residual_variances, expected_variances, rtol=1e-12)
    np.testing.assert_allclose(audit.assets.residual_vol**2, expected_variances, rtol=1e-12)
    centered_factors = result["factor_returns"]-result["factor_returns"].mean()
    reference_factors = 12*centered_factors.T @ centered_factors/len(centered_factors)
    np.testing.assert_allclose(inputs.factor_covariance, reference_factors, rtol=1e-12, atol=1e-14)
    reference = np.einsum("ik,kl,jl->ij", inputs.betas, reference_factors, inputs.betas)
    reference += np.diag(expected_variances)
    np.testing.assert_allclose(inputs.production_covariance, reference, rtol=1e-12, atol=1e-14)
    assert "fail" not in audit.validation["status"].values
    assert audit.provenance["residual_var_weight"] == .5


def test_connected_audit_uses_selected_residual_weight_without_changing_fit():
    """The bridge, GLS and audit consume the selected frozen residual covariance."""
    from factorlasso import VarianceColumns
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    result = namespace["inspect_fit"](replace(bundle, settings=replace(bundle.settings, residual_var_weight=.5)))
    snapshot, inputs, audit = result["snapshot"], result["inputs"], result["audit"]
    expected = .5*snapshot.data.y_variances[VarianceColumns.RESIDUAL_VARS.value]
    np.testing.assert_allclose(inputs.residual_variances, expected, rtol=1e-12)
    np.testing.assert_allclose(audit.assets.residual_vol**2, expected, rtol=1e-12)
    reference = np.einsum("ik,kl,jl->ij", inputs.betas, inputs.factor_covariance, inputs.betas)
    reference += np.diag(expected)
    np.testing.assert_allclose(inputs.production_covariance, reference, rtol=1e-12, atol=1e-14)
    assert audit.provenance["residual_var_weight"] == .5
