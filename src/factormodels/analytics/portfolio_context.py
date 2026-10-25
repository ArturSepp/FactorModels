"""Prepare generic portfolio-result inputs from a completed factor snapshot.

This module imports no portfolio optimiser. Its dictionary carries ordinary QIS
risk data and prepared pandas tables for consumers, including OP's result object.
The complete fitted model and statistical diagnostics remain owned by FM/FL.

Purpose: Prepare generic portfolio-result risk inputs from a completed model.
Inputs: A completed snapshot and the caller-selected asset order.
Outputs: prepare_portfolio_context supplies factor betas, groups and volatility context for reporting.
Units: Volatility is annual decimal; exposures retain declared native factor coordinates.
Conditioning: Use the snapshot selected covariance and residual policy without refitting or estimating alpha.
Ownership: FM prepares model context; QIS owns portfolio risk analytics and report rendering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factormodels.analytics.models import FactorModelSnapshot, PreparedSnapshotDiagnostics
from factormodels.analytics.risk import build_risk_model

__all__ = [
    'prepare_portfolio_context',
]


def prepare_portfolio_context(
    model: FactorModelSnapshot, *, diagnostics: PreparedSnapshotDiagnostics | None = None,
    diagonal_reference: bool = False,
) -> dict:
    """Prepare supplied risk and asset tables for a portfolio result constructor.

    Args:
        model: Completed single-date FM model, including its selected residual policy.
        diagnostics: Explicitly prepared historical-alpha display table. Omitted
            means span-free diagnostics only; this consumer never estimates alpha.
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
    total = model.asset_covariance
    residual = model.residual_covariance
    systematic = model.loadings @ model.factor_covariance @ model.loadings.T
    volatilities = pd.DataFrame({
        'total_vol': np.sqrt(np.diag(total)),
        'sys_vol': np.sqrt(np.diag(systematic)),
        'resid_vol': np.sqrt(np.diag(residual)),
    }, index=total.index)
    if diagnostics is not None:
        table = diagnostics.for_snapshot(model)
        table.loc[:, volatilities.columns] = volatilities
    elif model.fit_diagnostics is not None:
        table = model.fit_diagnostics
        table.loc[:, volatilities.columns] = volatilities
    else:
        table = pd.concat([
            model.loadings, model.fit_statistics.drop(columns=volatilities.columns, errors='ignore'),
            volatilities,
        ], axis=1)
    return {
        'risk_model': build_risk_model(model, diagonal_reference=diagonal_reference),
        'risk_model_date': model.cutoff,
        'decomposition_mode': 'diagonal_reference' if diagonal_reference else 'exact',
        'asset_diagnostics': table,
        'asset_volatilities': volatilities,
    }
