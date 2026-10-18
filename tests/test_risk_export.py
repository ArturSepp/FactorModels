"""Risk interchange preserves selected total covariance and native coordinates."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis
from factorlasso import LassoModel, LassoModelType
from qis.perfstats.config import ReturnTypes
from qis.portfolio.risk.factor_groups import FactorGroupSpec

from tradablefactors.estimation import FactorCovarEstimator
from tradablefactors.model import DatedFactorCovariance
from tradablefactors.risk import build_risk_model
from tradablefactors.tests._estimation_fixture import estimator, panels


@pytest.mark.parametrize("policy", ["orthogonal", "empirical", "residual_cluster"])
def test_export_preserves_selected_sigma_and_supported_decomposition(policy):
    """QIS risk uses full selected covariance even when its factor view is limited."""
    factors, returns = panels()
    snapshot = estimator(residual_type=policy, residual_var_weight=.6).fit_current(
        factors, returns, cutoff=factors.index[-1])
    model = snapshot.to_risk_model()
    pd.testing.assert_frame_equal(model.covar[snapshot.cutoff], snapshot.asset_covariance)
    pd.testing.assert_frame_equal(model.factor_loadings[snapshot.cutoff], snapshot.data.y_betas)
    correlated = np.any(snapshot.residual_covariance.to_numpy() != np.diag(np.diag(snapshot.residual_covariance)))
    if correlated:
        assert model.factor_covar is None and model.residual_vars is None
        reference = snapshot.to_risk_model(diagonal_reference=True)
        assert reference.factor_covar is not None
        pd.testing.assert_frame_equal(reference.covar[snapshot.cutoff], snapshot.asset_covariance)
    else:
        np.testing.assert_allclose(model.residual_vars[snapshot.cutoff], np.diag(snapshot.residual_covariance))
    portfolio = pd.Series([.4, .35, .25], index=snapshot.data.y_betas.index)
    benchmark = pd.Series(1. / 3., index=portfolio.index)
    manual = qis.RiskModel(covar={snapshot.cutoff: snapshot.asset_covariance})
    assert model.compute_tre_at_date(benchmark, portfolio, snapshot.cutoff) == pytest.approx(
        manual.compute_tre_at_date(benchmark, portfolio, snapshot.cutoff), rel=1e-13)


def test_plain_covariance_and_explicit_raw_fl_policies():
    """A raw FL container has no selected policy and cannot silently choose one."""
    factors, returns = panels()
    snapshot = estimator(residual_type="empirical").fit_current(factors, returns, cutoff=factors.index[-1])
    raw = {snapshot.cutoff: snapshot.data}
    with pytest.raises(ValueError, match="explicit residual_type"):
        build_risk_model(raw)
    model = build_risk_model(raw, residual_type="empirical")
    pd.testing.assert_frame_equal(model.covar[snapshot.cutoff], snapshot.asset_covariance)
    plain = build_risk_model({snapshot.cutoff: snapshot.asset_covariance})
    assert plain.factor_loadings is None
    with pytest.raises(ValueError, match="already carry"):
        build_risk_model(snapshot, residual_type="orthogonal")
    with pytest.raises(ValueError, match="empty"):
        build_risk_model({})


@pytest.mark.parametrize("options", [
    {"residual_var_weight": .5},
    {"residual_corr_weight": .5},
    {"diagonal_reference": True},
])
def test_plain_covariance_rejects_unrepresentable_policy_overrides(options):
    """A supplied Sigma cannot appear to change its residual risk without a decomposition."""
    covariance = pd.DataFrame([[.04]], index=["asset"], columns=["asset"])
    with pytest.raises(ValueError, match="covariance-only"):
        build_risk_model({pd.Timestamp("2024-01-31"): covariance}, **options)


def test_raw_factor_axes_cannot_silently_use_positional_covariance():
    """Reversed factor labels must not produce a plausible but incorrectly paired Sigma."""
    factors, returns = panels()
    snapshot = estimator().fit_current(factors, returns, cutoff=factors.index[-1])
    bad = replace(snapshot.data, x_covar=snapshot.data.x_covar.iloc[::-1, ::-1])
    with pytest.raises(ValueError, match="same order"):
        build_risk_model({snapshot.cutoff: bad}, residual_type="orthogonal")


def test_native_coordinates_and_group_metadata_survive_export():
    """Point-valued factor loadings retain their units all the way to RiskModel."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2018-01-31", periods=80, freq="ME")
    levels = pd.DataFrame({"VIX": rng.uniform(12, 45, len(dates))}, index=dates)
    slopes = np.array([-.002, .001, -.004])
    returns = pd.DataFrame(levels.VIX.diff().to_numpy()[:, None] * slopes,
                           index=dates, columns=["a", "b", "c"])
    groups = {"volatility": FactorGroupSpec("volatility", ("VIX",))}
    fitted = FactorCovarEstimator(
        lasso_model=LassoModel(model_type=LassoModelType.LASSO, reg_lambda=0.,
                               demean=False, solver="CLARABEL", span=24, warmup_period=12),
        factor_returns_freq="ME", factor_covar_span=24,
        factor_return_types={"VIX": ReturnTypes.DIFFERENCE}, factor_groups=groups,
    ).fit_current(levels, {"ME": returns}, cutoff=dates[-1])
    np.testing.assert_allclose(fitted.data.y_betas.VIX, slopes, atol=1e-7)
    model = fitted.to_risk_model()
    assert model.factor_return_types == {"VIX": ReturnTypes.DIFFERENCE}
    assert model.factor_groups == groups
    assert fitted.data.x_covar.iloc[0, 0] > 100.
    wrong_units = DatedFactorCovariance(fitted.data.x_covar, dates[-1], dates[-1])
    with pytest.raises(ValueError, match="return types"):
        wrong_units.at_cutoff(dates[-1], levels.columns, fitted.factor_return_types)


def test_rolling_export_preserves_each_selected_covariance():
    """Multiple residual snapshots retain their individual total risk on export."""
    factors, returns = panels()
    rolling = estimator(residual_type="empirical").fit_rolling(
        factors, returns, time_period=qis.TimePeriod("2022-09-30", "2022-12-31"))
    model = rolling.to_risk_model()
    for date, covariance in rolling.covariances.items():
        pd.testing.assert_frame_equal(model.covar[date], covariance)
