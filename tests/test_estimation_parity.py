"""Acceptance checks for the completed current/rolling estimation product."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis

from tradablefactors.tests._estimation_fixture import estimator, panels


def test_factor_covariance_sampling_and_annualisation_match_explicit_weights():
    """Check sampling, mean removal and annual units without the QIS covariance kernel."""
    factors, returns = panels()
    fitted = estimator().fit_current(factors, returns, cutoff=factors.index[-1])
    sampled = factors.resample("ME").last()
    log_returns = np.log(sampled).diff().iloc[1:]
    deviations = (log_returns - log_returns.ewm(span=36, adjust=False).mean()).iloc[1:]
    decay = 1. - 2. / 37.
    weights = (1. - decay) * decay ** np.arange(len(deviations) - 1, -1, -1)
    reference = 12. * np.einsum("t,ti,tj->ij", weights, deviations, deviations)
    np.testing.assert_allclose(fitted.data.x_covar, reference, rtol=1e-12, atol=1e-14)


def test_completed_covariance_matches_independent_factor_pair_sum():
    """Rebuild every covariance entry independently of FL assembly."""
    factors, returns = panels()
    fitted = estimator(residual_var_weight=.35).fit_current(
        factors, returns, cutoff=factors.index[-1])
    beta, factor = fitted.data.y_betas, fitted.data.x_covar
    residual = fitted.residual_covariance
    reference = np.empty((len(beta), len(beta)))
    for i, a in enumerate(beta.index):
        for j, b in enumerate(beta.index):
            reference[i, j] = sum(
                beta.at[a, k] * factor.at[k, ell] * beta.at[b, ell]
                for k in beta.columns for ell in beta.columns
            ) + residual.at[a, b]
    np.testing.assert_allclose(fitted.asset_covariance, reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.diag(residual), .35 * fitted.data.y_variances.residual_var, rtol=1e-14)
    assert fitted.data.residual_metadata.frequency.to_dict() == {"a": "ME", "b": "ME", "c": "QE"}
    assert fitted.data.residual_metadata.annualisation_factor.to_dict() == {"a": 12., "b": 12., "c": 4.}


def test_empirical_result_owns_selected_full_residual_risk():
    """Reconstruct selected correlated covariance directly from prepared correlation."""
    factors, returns = panels()
    fitted = estimator(residual_type="empirical", residual_corr_weight=.6).fit_current(
        factors, returns, cutoff=factors.index[-1])
    variances = fitted.data.y_variances.residual_var.to_numpy()
    corr = fitted.data.residual_correlation.correlation.to_numpy()
    reference = .6 * corr * np.sqrt(variances[:, None] * variances[None, :])
    np.fill_diagonal(reference, variances)
    np.testing.assert_allclose(fitted.residual_covariance, reference, rtol=1e-13, atol=1e-15)
    assert np.any(np.abs(reference - np.diag(variances)) > 1e-7)
    beta, factor = fitted.data.y_betas.to_numpy(), fitted.data.x_covar.to_numpy()
    np.testing.assert_allclose(fitted.asset_covariance, beta @ factor @ beta.T + reference)


@pytest.mark.parametrize("policy", ["orthogonal", "empirical"])
def test_current_and_rolling_results_have_the_same_explicit_date_contract(policy):
    """Current B/F/diagonals agree; rolling empirical correlation is held between updates."""
    factors, returns = panels()
    configured = estimator(residual_type=policy)
    period = qis.TimePeriod("2022-06-30", "2022-12-31")
    rolling = configured.fit_rolling(factors, returns, time_period=period)
    assert len(rolling.snapshots) > 1
    for date, snapshot in rolling.snapshots.items():
        current = configured.fit_current(factors, returns, cutoff=date)
        pd.testing.assert_frame_equal(snapshot.data.y_betas, current.data.y_betas)
        pd.testing.assert_frame_equal(snapshot.data.x_covar, current.data.x_covar)
        np.testing.assert_allclose(np.diag(snapshot.asset_covariance), np.diag(current.asset_covariance))
        if policy == "orthogonal" or snapshot.data.residual_correlation.estimation_date == date:
            pd.testing.assert_frame_equal(snapshot.asset_covariance, current.asset_covariance)
        else:
            assert snapshot.data.residual_correlation.estimation_date < date
            assert snapshot.data.residual_correlation.observation_date <= date
        assert snapshot.cutoff == date
    assert len(rolling.data.data) == len(rolling.covariances)


def test_fits_do_not_mutate_model_configuration_or_previous_result():
    """Fitted state belongs to each completed result, not the supplied FL config."""
    factors, returns = panels()
    configured = estimator()
    assert configured.lasso_model.estimated_betas is None
    first = configured.fit_current(factors, returns, cutoff=factors.index[-12])
    expected = first.data.y_betas.copy()
    second = configured.fit_current(factors, returns, cutoff=factors.index[-1])
    assert configured.lasso_model.estimated_betas is None
    pd.testing.assert_frame_equal(first.data.y_betas, expected)
    assert first.data is not second.data
    changed = replace(configured, residual_var_weight=0.)
    factor_only = changed.fit_current(factors, returns, cutoff=factors.index[-12])
    np.testing.assert_allclose(factor_only.residual_covariance, 0.)
