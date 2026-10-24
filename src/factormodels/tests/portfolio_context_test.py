"""Prepared result context preserves selected risk without importing an optimiser."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData, ResidualCorrelationData
from qis.perfstats.config import ReturnTypes

from factormodels.estimation.models import FactorModelSnapshot
from factormodels.analytics.portfolio_context import prepare_portfolio_context


def _snapshot(complete: bool = True, correlated: bool = False) -> FactorModelSnapshot:
    """Construct a small fitted result with known native coordinates and residual risk."""
    date = pd.Timestamp('2026-07-31')
    assets = pd.Index(['one', 'two'])
    variances = pd.DataFrame({'residual_var': [0.01, 0.04]}, index=assets)
    if complete:
        variances['r2'] = [0.5, 0.6]
        variances['insample_alpha'] = [0.03, 0.04]
    correlation = None
    if correlated:
        correlation = ResidualCorrelationData(
            correlation=pd.DataFrame([[1., 0.4], [0.4, 1.]], index=assets, columns=assets),
            residual_returns=pd.DataFrame([[0., 0.]], index=[date], columns=assets),
            asset_metadata=pd.DataFrame(index=assets), frequency='ME', span=12,
            observation_date=date, estimation_date=date)
    data = CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.16]], index=['yield'], columns=['yield']),
        y_betas=pd.DataFrame([-0.1, -0.2], index=assets, columns=['yield']),
        y_variances=variances, estimation_date=date, residual_correlation=correlation)
    return FactorModelSnapshot(
        data=data, cutoff=date, residual_type='empirical' if correlated else 'orthogonal',
        factor_return_types={'yield': ReturnTypes.DIFFERENCE})


@pytest.mark.parametrize('complete', [False, True])
def test_prepared_context_retains_fitted_diagnostics_and_native_coordinates(complete: bool) -> None:
    """Complete and minimal fits both produce ordinary independently owned report tables."""
    model = _snapshot(complete=complete)
    context = prepare_portfolio_context(model)
    risk = context['risk_model']
    assert context['risk_model_date'] == model.cutoff
    assert context['decomposition_mode'] == 'exact'
    assert risk.factor_return_types == {'yield': ReturnTypes.DIFFERENCE}
    pd.testing.assert_frame_equal(risk.covar[model.cutoff], model.asset_covariance)
    np.testing.assert_allclose(context['asset_volatilities']['total_vol'] ** 2,
                               np.diag(model.asset_covariance))
    if complete:
        pd.testing.assert_frame_equal(context['asset_diagnostics'], model.data.get_snapshot())
    context['asset_diagnostics'].iloc[0, 0] = 999
    assert model.data.y_betas.iloc[0, 0] == -0.1


def test_full_residual_policy_and_variance_weight_control_prepared_tables() -> None:
    """Report context uses the selected policy, including a nondefault variance multiplier."""
    model = replace(_snapshot(correlated=True), residual_var_weight=0.3)
    context = prepare_portfolio_context(model)
    risk = context['risk_model']
    assert risk.factor_covar is None
    pd.testing.assert_frame_equal(risk.covar[model.cutoff], model.asset_covariance)
    np.testing.assert_allclose(context['asset_diagnostics']['resid_vol'] ** 2,
                               [0.003, 0.012])
    reference = prepare_portfolio_context(model, diagonal_reference=True)
    assert reference['decomposition_mode'] == 'diagonal_reference'
    assert reference['risk_model'].factor_covar is not None
    pd.testing.assert_frame_equal(reference['risk_model'].covar[model.cutoff],
                                   model.asset_covariance)


def test_context_requires_a_completed_snapshot() -> None:
    """Unselected raw FL data cannot silently choose a downstream residual policy."""
    with pytest.raises(TypeError, match='completed FactorModelSnapshot'):
        prepare_portfolio_context(_snapshot().data)
