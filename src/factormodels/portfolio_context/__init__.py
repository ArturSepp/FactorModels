"""Prepare generic portfolio-result inputs from a completed factor snapshot.

This module imports no portfolio optimiser. Its dictionary carries ordinary QIS
risk data and prepared pandas tables for consumers, including OP's result object.
The complete fitted model and statistical diagnostics remain owned by FM/FL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factormodels.model import FactorModelSnapshot
from factormodels.risk import build_risk_model

__all__ = ['prepare_portfolio_context']


def prepare_portfolio_context(
    model: FactorModelSnapshot, *, diagonal_reference: bool = False,
) -> dict:
    """Prepare supplied risk and asset tables for a portfolio result constructor.

    Args:
        model: Completed single-date FM model, including its selected residual policy.
        diagonal_reference: Include and explicitly label the diagonal F/D reference.
            Full covariance remains authoritative for total risk and tracking error.

    Returns:
        Keyword inputs ``risk_model``, ``risk_model_date``, ``decomposition_mode``,
        ``asset_diagnostics`` and ``asset_volatilities``. The tables retain native
        factor units. Volatilities use the selected covariance and residual policy.
        Pass these alongside separately supplied portfolio and benchmark weights.
    """
    if not isinstance(model, FactorModelSnapshot):
        raise TypeError('model must be a completed FactorModelSnapshot')
    data = model.data
    total = model.asset_covariance
    residual = model.residual_covariance
    systematic = data.y_betas @ data.x_covar @ data.y_betas.T
    volatilities = pd.DataFrame({
        'total_vol': np.sqrt(np.diag(total)),
        'sys_vol': np.sqrt(np.diag(systematic)),
        'resid_vol': np.sqrt(np.diag(residual)),
    }, index=total.index)
    if {'r2', 'insample_alpha'}.issubset(data.y_variances.columns):
        diagnostics = data.get_snapshot().copy()
        diagnostics.loc[:, volatilities.columns] = volatilities
    else:
        diagnostics = pd.concat([
            data.y_betas, data.y_variances.drop(columns=volatilities.columns, errors='ignore'),
            volatilities,
        ], axis=1)
    return {
        'risk_model': build_risk_model(model, diagonal_reference=diagonal_reference),
        'risk_model_date': model.cutoff,
        'decomposition_mode': 'diagonal_reference' if diagonal_reference else 'exact',
        'asset_diagnostics': diagnostics,
        'asset_volatilities': volatilities,
    }
