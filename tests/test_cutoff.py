"""Point-in-time contracts and deliberate counterexamples for all fit inputs."""
import numpy as np
import pandas as pd
import pytest
import qis

from factormodels.model import DatedFactorCovariance
from factormodels.tests._estimation_fixture import estimator, panels


@pytest.mark.parametrize("policy", ["orthogonal", "empirical", "residual_cluster"])
def test_future_observations_cannot_change_current_model(policy):
    """Changing all post-cutoff factors/assets cannot influence the dated fit."""
    factors, returns = panels()
    cutoff = factors.index[-13]
    reference = estimator(residual_type=policy).fit_current(factors, returns, cutoff=cutoff)
    changed_factors = factors.copy()
    changed_factors.loc[changed_factors.index > cutoff] *= 100.
    changed_returns = {freq: panel.copy() for freq, panel in returns.items()}
    for panel in changed_returns.values():
        panel.loc[panel.index > cutoff] = 9.
    changed = estimator(residual_type=policy).fit_current(changed_factors, changed_returns, cutoff=cutoff)
    pd.testing.assert_frame_equal(reference.asset_covariance, changed.asset_covariance)
    pd.testing.assert_frame_equal(reference.data.residuals, changed.data.residuals)
    assert reference.data.residuals.index.max() <= cutoff
    assert all(date <= cutoff for date in reference.provenance["asset_observation_dates"].values())


def test_rolling_fit_does_not_use_future_periods():
    """Later factor/asset rows cannot alter earlier scheduled snapshots."""
    factors, returns = panels()
    cutoff = factors.index[-13]
    period = qis.TimePeriod("2021-06-30", cutoff)
    configured = estimator(residual_type="empirical")
    actual = configured.fit_rolling(factors, returns, time_period=period)
    reference = configured.fit_rolling(
        factors.loc[:cutoff], {freq: panel.loc[:cutoff] for freq, panel in returns.items()},
        time_period=period)
    for date in reference.snapshots:
        pd.testing.assert_frame_equal(actual.covariances[date], reference.covariances[date])


def test_supplied_covariance_provenance_is_checked_before_fit():
    """A dated supplied covariance cannot hide a future observation or publication."""
    factors, returns = panels()
    cutoff = factors.index[-13]
    matrix = pd.DataFrame(np.diag([.04, .03]), index=factors.columns, columns=factors.columns)
    supplied = DatedFactorCovariance(matrix, cutoff, cutoff)
    fitted = estimator().fit_current(factors, returns, cutoff=cutoff, factor_covariance=supplied)
    pd.testing.assert_frame_equal(fitted.data.x_covar, matrix)
    with pytest.raises(ValueError, match="available at cutoff"):
        estimator().fit_current(
            factors, returns, cutoff=cutoff,
            factor_covariance=DatedFactorCovariance(matrix, cutoff, cutoff + pd.Timedelta(days=1)))
    with pytest.raises(ValueError, match="observation_date"):
        DatedFactorCovariance(matrix, cutoff + pd.Timedelta(days=1), cutoff)
    with pytest.raises(TypeError, match="DatedFactorCovariance"):
        estimator().fit_current(factors, returns, cutoff=cutoff, factor_covariance=matrix)


def test_supplied_cluster_maps_need_dated_provenance():
    """Externally prepared partitions cannot bypass the cutoff contract."""
    factors, returns = panels()
    cutoff = factors.index[-13]
    with pytest.raises(ValueError, match="precomputed_asof"):
        estimator().fit_current(factors, returns, cutoff=cutoff, precomputed_clusters={})
    with pytest.raises(ValueError, match="available at cutoff"):
        estimator().fit_current(factors, returns, cutoff=cutoff, precomputed_clusters={},
                                precomputed_asof=factors.index[-1])


def test_invalid_histories_are_rejected_before_numerical_work():
    """Ambiguous observation ordering and duplicated asset buckets fail clearly."""
    factors, returns = panels()
    with pytest.raises(ValueError, match="ordered unique"):
        estimator().fit_current(factors.iloc[::-1], returns, cutoff=factors.index[-1])
    with pytest.raises(ValueError, match="exactly one"):
        estimator().fit_current(factors, {"ME": returns["ME"], "QE": returns["ME"]},
                                cutoff=factors.index[-1])
    with pytest.raises(ValueError, match="no observations"):
        estimator().fit_current(factors, returns, cutoff=pd.Timestamp("1900-01-01"))
