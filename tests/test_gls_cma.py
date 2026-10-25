"""Completed premia/CMA outputs checked with independent numerical constructions."""

import numpy as np
import pandas as pd

from factormodels.gls_diagnostic import CmaDiagnosticInputs, compute_cma_diagnostics
from factormodels.gls_diagnostic import infer_factor_premia


def test_premia_cash_and_alpha_match_independent_weighted_qr():
    """Cross-sectional preparation subtracts cash and admitted alpha exactly once."""
    assets, factors = list('abcde'), ['Equity', 'Rates']
    betas = pd.DataFrame([[1., .2], [.4, 1.], [-.3, .8], [.9, -.2], [.6, .7]],
                         index=assets, columns=factors)
    residual = pd.Series([.01, .02, .03, .015, .025], index=assets)
    forecasts = pd.Series([.09, .06, .04, .08, .055], index=assets)
    alpha = pd.Series([.003, 0., -.001, .004, .002], index=assets)
    result = infer_factor_premia(betas, residual, forecasts,
                                cash_rate=.03, alpha=alpha, alpha_weight=.5)
    target = forecasts - .03 - .5 * alpha
    scale = np.sqrt(residual.to_numpy())
    q, r = np.linalg.qr(betas.to_numpy() / scale[:, None])
    independent = np.linalg.solve(r, q.T @ (target.to_numpy() / scale))
    np.testing.assert_allclose(result.premia, independent, atol=1e-12, rtol=1e-10)
    assert result.premia.index.tolist() == factors
    np.testing.assert_allclose(result.assets.fitted + result.assets.residual
                               + result.assets.cash + result.assets.credited_alpha,
                               forecasts, atol=1e-12, rtol=1e-10)


def test_cma_completed_tables_reconcile_with_direct_asset_covariance():
    """The fitted-plus-residual Sharpe accounting agrees with an asset-space solve."""
    assets, factors = list('abcd'), ['Equity', 'Rates']
    betas = pd.DataFrame([[1., 0.], [0., 1.], [.5, .6], [1., -.2]],
                         index=assets, columns=factors)
    residual = pd.Series([.01, .02, .03, .04], index=assets)
    covariance = pd.DataFrame([[.04, .005], [.005, .01]], index=factors, columns=factors)
    premia = pd.Series([.05, .02], index=factors)
    forecast = betas @ premia + pd.Series([.01, .015, -.01, .003], index=assets)
    result = compute_cma_diagnostics(
        CmaDiagnosticInputs(betas, residual, forecast, covariance, premia))
    asset_covariance = betas.to_numpy() @ covariance.to_numpy() @ betas.to_numpy().T
    asset_covariance += np.diag(residual)
    direct = float(forecast @ np.linalg.solve(asset_covariance, forecast))
    np.testing.assert_allclose(result.summary['total_sharpe2'], direct,
                               atol=1e-12, rtol=1e-10)
    assert result.validation.status.eq('pass').all()
    assert result.assets.index.tolist() == assets
    assert result.factors.index.tolist() == factors
