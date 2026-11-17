"""Native alpha and common-grid empirical residual covariance integration."""

import numpy as np
import pandas as pd
import pytest
import qis
import factorlasso as fl

import factormodels.estimation._pipeline as module
from factormodels import FactorCovarEstimator, estimate_historical_alpha



@pytest.fixture
def panels():
    """Complete synthetic monthly/quarterly log returns with correlated residuals."""
    rng = np.random.default_rng(2092026)
    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    x = pd.DataFrame(rng.normal(0.002, 0.035, (108, 2)), index=dates, columns=["f", "g"])
    prices = np.exp(x.cumsum())
    m = pd.DataFrame(x.to_numpy() @ np.array([[0.7, 0.2], [0.1, -0.3]])
                     + rng.normal(0.001, 0.02, (108, 2)), index=dates, columns=["a", "b"])
    q = pd.DataFrame({"c": (0.4 * x.f + rng.normal(0., 0.02, 108)).resample("QE").sum()})
    return prices, {"ME": m, "QE": q}


def estimator(**kwargs):
    """Use the native model's per-frequency spans and an explicit empirical opt-in."""
    options = dict(lasso_model=fl.LassoModel(span=36, span_freq_dict={"ME": 36, "QE": 12},
                                           warmup_period=6, reg_lambda=1e-5),
                   factor_returns_freq="ME", rebalancing_freq="ME", residual_type="empirical")
    options.update(kwargs)
    return FactorCovarEstimator(**options)


def test_current_prepares_annual_covariance_and_preserves_alpha(panels):
    """Preparation changes no fitted beta, annual-alpha series or orthogonal matrix."""
    prices, returns = panels
    empirical = estimator().fit_current(prices, returns, cutoff=prices.index[-1])
    legacy = estimator(residual_type="orthogonal").fit_current(prices, returns, cutoff=prices.index[-1])
    pd.testing.assert_frame_equal(empirical.residuals, legacy.residuals)
    pd.testing.assert_series_equal(estimate_historical_alpha(empirical), estimate_historical_alpha(legacy))
    pd.testing.assert_frame_equal(empirical.raw_diagonal_reference_covariance, legacy.asset_covariance)
    assert empirical.residual_correlation_metadata['frequency'] == "QE"
    assert empirical.residual_correlation_metadata['span'] == pytest.approx(12)
    assert empirical.residual_metadata.frequency.to_dict() == {"a": "ME", "b": "ME", "c": "QE"}
    assert empirical.residual_metadata.residual_scale.to_dict() == {"a": 12., "b": 12., "c": 4.}
    assembled = empirical.asset_covariance
    v = empirical.fit_statistics.residual_var.to_numpy()
    r = empirical.residual_correlation.to_numpy()
    d = r * np.sqrt(v[:, None] * v[None, :])
    np.fill_diagonal(d, v)
    expected = (empirical.loadings @ empirical.factor_covariance @ empirical.loadings.T) + d
    np.testing.assert_allclose(assembled, expected, rtol=1e-14)
    np.testing.assert_array_equal(np.diag(assembled), np.diag(legacy.asset_covariance))
    assert empirical.residual_type == "empirical"


def test_rolling_holds_residual_estimate_between_quarters(panels):
    """New monthly betas must not refresh a covariance without a new complete quarter."""
    prices, returns = panels
    period = qis.TimePeriod("2021-06-30", "2022-06-30")
    result = estimator().fit_rolling(prices, returns, time_period=period)
    first = result.snapshots[pd.Timestamp("2021-06-30")]
    for date in pd.to_datetime(["2021-07-31", "2021-08-31"]):
        held = result.snapshots[date]
        pd.testing.assert_frame_equal(held.residual_correlation, first.residual_correlation)
        assert held.residual_correlation_metadata['observation_date'] == pd.Timestamp("2021-06-30")
    september = result.snapshots[pd.Timestamp("2021-09-30")]
    assert september.residual_correlation_metadata['observation_date'] == pd.Timestamp("2021-09-30")
    for date, fitted in result.snapshots.items():
        info = fitted.residual_correlation_metadata
        assert info['estimation_date'] <= date
        assert info['observation_date'] <= date
        assert info['observation_date'].month % 3 == 0
        pd.testing.assert_frame_equal(result.covariances[date], fitted.asset_covariance)



def test_empirical_current_fit_truncates_inputs(panels):
    """The empirical opt-in cannot use later beta or residual information."""
    prices, returns = panels
    cutoff = pd.Timestamp("2021-05-31")
    full = estimator().fit_current(prices, returns, cutoff=cutoff)
    truncated = estimator().fit_current(
        prices.loc[:cutoff], {freq: data.loc[:cutoff] for freq, data in returns.items()},
        cutoff=cutoff,
    )
    pd.testing.assert_frame_equal(full.asset_covariance,
                                  truncated.asset_covariance)
    assert full.residual_correlation_metadata['observation_date'] == pd.Timestamp("2021-03-31")
    assert full.residual_correlation_metadata['estimation_date'] == cutoff
    assert full.cutoff == cutoff


def test_explicit_grid_and_span(panels):
    """A caller can prepare a common grid explicitly with a span in that grid's units."""
    prices, returns = panels
    result = estimator(residual_covar_freq="QE", residual_covar_span=8).fit_current(
        prices, {"ME": returns["ME"]}, cutoff=prices.index[-1],
    )
    assert result.residual_correlation_metadata['frequency'] == "QE"
    assert result.residual_correlation_metadata['span'] == 8
    assert result.residual_correlation_metadata['schema_version'] == 2
    assert result.residual_correlation_metadata['asset_metadata'].annualisation_factor.eq(12).all()



def test_invalid_selection(panels):
    """Reject unknown residual structures instead of silently selecting a default."""
    with pytest.raises(ValueError, match="residual_type"):
        estimator(residual_type="joint")
    prices, returns = panels
    with pytest.raises(ValueError, match="residual_type"):
        module._estimate_factor_covar_data(prices, returns, fl.LassoModel(),
                                               residual_type="joint")


def test_retention_forwarding_and_refreshing_diagonal(panels):
    """Both estimator interfaces forward rho; held quarterly R uses each month's variance."""
    prices, returns = panels
    model = estimator(residual_corr_weight=.5)
    current = model.fit_current(prices, returns, cutoff=prices.index[-1])
    assert current.residual_corr_weight == .5
    period = qis.TimePeriod("2021-06-30", "2021-08-31")
    rolling = model.fit_rolling(prices, returns, time_period=period)
    observations = {fit.residual_correlation_metadata['observation_date']
                    for fit in rolling.snapshots.values()}
    assert len(observations) == 1 and len(rolling.snapshots) == 3
    previous = None
    for date, fitted in rolling.snapshots.items():
        d = fitted.residual_covariance
        variances = fitted.fit_statistics.residual_var.to_numpy()
        expected = .5 * fitted.residual_correlation.to_numpy() * np.sqrt(
            variances[:, None] * variances[None, :])
        np.fill_diagonal(expected, variances)
        np.testing.assert_allclose(d, expected, rtol=1e-14)
        np.testing.assert_array_equal(np.diag(d), variances)
        if previous is not None:
            assert not np.allclose(d, previous)
        previous = d
        pd.testing.assert_frame_equal(rolling.covariances[date], fitted.asset_covariance)



@pytest.mark.parametrize("weight", [-.1, 1.1, np.nan, np.inf])
def test_invalid_retention(weight):
    """Reject invalid correlation mixtures before fitting."""
    with pytest.raises(ValueError, match="residual_corr_weight"):
        estimator(residual_corr_weight=weight)


def test_orthogonal_retention_cannot_be_ignored():
    """A nondefault empirical-only setting cannot silently select the orthogonal model."""
    with pytest.raises(ValueError, match="residual_corr_weight"):
        estimator(residual_type="orthogonal", residual_corr_weight=.5)


def test_empirical_fit_keeps_unfitted_zero_risk_asset_independent(panels):
    """An output-only asset has no residual covariance and cannot block a valid fit."""
    prices, returns = panels
    fitted = estimator().fit_current(prices, returns, cutoff=prices.index[-1])
    expanded = estimator().fit_current(
        prices, returns, cutoff=prices.index[-1], assets=["a", "b", "c", "future"]
    )
    corr = expanded.residual_correlation
    assert corr.index.tolist() == ["a", "b", "c", "future"]
    pd.testing.assert_frame_equal(corr.loc[["a", "b", "c"], ["a", "b", "c"]],
                                  fitted.residual_correlation)
    assert corr.loc["future", "future"] == 1.0
    assert corr.loc["future", ["a", "b", "c"]].eq(0).all()
    covariance = expanded.asset_covariance
    pd.testing.assert_frame_equal(covariance.loc[["a", "b", "c"], ["a", "b", "c"]],
                                  fitted.asset_covariance)
    assert covariance.loc["future"].eq(0).all()


def test_empirical_fit_rejects_universe_without_positive_residual_risk(panels):
    """A wholly unfitted output universe cannot define empirical correlation."""
    prices, returns = panels
    with pytest.raises(ValueError, match="positive residual risk"):
        estimator().fit_current(prices, returns, cutoff=prices.index[-1], assets=["future"])


@pytest.mark.parametrize("kind", ["exposure_cluster", "residual_cluster"])
def test_cluster_policies_prepare_causally_and_forward_assembly(panels, kind):
    """Both cluster policies share empirical preparation and retain the chosen assembly."""
    prices, returns = panels
    # A one-asset cadence falls back to plain LASSO and has no exposure partition.
    # Supply two assets per cadence to exercise the cluster model on both grids.
    quarterly = returns['QE'].copy()
    quarterly['d'] = .5 * quarterly['c'] + .003 * np.sin(np.arange(len(quarterly)))
    returns = {**returns, 'QE': quarterly}
    lasso = fl.LassoModel(model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                         span=36, span_freq_dict={"ME": 36, "QE": 12},
                         warmup_period=6, reg_lambda=1e-5)
    model = estimator(lasso_model=lasso, residual_type=kind, residual_corr_weight=.5)
    cutoff = pd.Timestamp("2021-05-31")
    current = model.fit_current(prices, returns, cutoff=cutoff)
    empirical = estimator(lasso_model=lasso.copy(kwargs={})).fit_current(
        prices, returns, cutoff=cutoff)
    pd.testing.assert_frame_equal(current.loadings, empirical.loadings)
    pd.testing.assert_frame_equal(current.residual_correlation,
                                  empirical.residual_correlation)
    assert current.residual_correlation_metadata['observation_date'] <= cutoff
    assert current.residual_type == kind
    assert current.residual_corr_weight == .5
    # FL is the assembly owner; reconstruct its public input independently of FM's adapter.
    correlation = fl.ResidualCorrelationData(
        correlation=current.residual_correlation,
        **{name: value for name, value in current.residual_correlation_metadata.items()
           if name != 'schema_version'},
    )
    raw = fl.CurrentFactorCovarData(
        x_covar=current.factor_covariance, y_betas=current.loadings,
        y_variances=current.fit_statistics, estimation_date=cutoff,
        clusters=current.clusters, residual_correlation=correlation,
    )
    expected = raw.get_residual_covar(residual_type=kind, residual_corr_weight=.5)
    pd.testing.assert_frame_equal(current.residual_covariance, expected)
    period = qis.TimePeriod("2021-06-30", "2021-08-31")
    rolling = model.fit_rolling(prices, returns, time_period=period)
    assert len({fit.residual_correlation_metadata['observation_date']
                for fit in rolling.snapshots.values()}) == 1
    for date, fitted in rolling.snapshots.items():
        assert fitted.residual_type == kind
        assert fitted.residual_corr_weight == .5
        pd.testing.assert_frame_equal(rolling.covariances[date], fitted.asset_covariance)



@pytest.mark.parametrize('residual_type', ['orthogonal', 'empirical'])
@pytest.mark.parametrize('factor_start', [0, 36, 84])
def test_residuals_preserve_available_history_before_factor_inception(
        panels, residual_type, factor_start):
    """An unavailable factor must not erase observed responses or other factor contributions."""
    prices, returns = panels
    prices = prices.copy()
    prices.iloc[:factor_start, prices.columns.get_loc('g')] = np.nan
    cutoff = prices.index[71]
    prices = prices.loc[:cutoff]
    monthly = returns['ME'].loc[:cutoff].copy()
    monthly.loc[monthly.index[:3], 'a'] = np.nan
    fitted = estimator(residual_type=residual_type).fit_current(
        prices, {'ME': monthly}, cutoff=cutoff,
    )

    # Independently sum only observed contributions for each date and response.
    # A wholly missing factor row is still unavailable; missing responses stay missing.
    factor_returns = np.log(prices / prices.shift(1))
    expected = monthly.copy()
    for date, observations in factor_returns.iterrows():
        observed = observations.dropna()
        for asset in expected.columns:
            expected.loc[date, asset] = (
                monthly.loc[date, asset]
                - sum(value * fitted.loadings.loc[asset, factor]
                      for factor, value in observed.items())
                if not observed.empty else np.nan
            )
    pd.testing.assert_frame_equal(fitted.residuals / 12., expected, rtol=1e-12, atol=1e-14)
    assert fitted.residuals.loc[monthly.a.isna(), 'a'].isna().all()
    assert fitted.fit_statistics.residual_var.gt(0).all()
    if factor_start == 0:
        legacy = 12. * (monthly - factor_returns @ fitted.loadings.T)
        pd.testing.assert_frame_equal(fitted.residuals, legacy, rtol=1e-12, atol=1e-14)
    if residual_type == 'empirical':
        assert np.isfinite(fitted.residual_correlation.to_numpy()).all()
        covariance = fitted.asset_covariance
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12
