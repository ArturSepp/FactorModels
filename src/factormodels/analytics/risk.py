"""Export selected FM risk to the existing QIS RiskModel contract.
Purpose: Export completed FM covariance through the QIS RiskModel interface.
Inputs: A snapshot, rolling snapshots or a dated mapping of supplied asset covariances.
Outputs: build_risk_model returns a QIS risk view retaining the exact selected Sigma.
Units: Covariance is annual squared return in the declared native factor coordinates.
Conditioning: The optional diagonal reference uses diag(selected R); it never replaces correlated selected Sigma.
Ownership: Analytics performs interchange only; fitting and raw-FL adaptation belong to estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis

from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel


def build_risk_model(
    model: FactorModelSnapshot | RollingFactorModel | dict[pd.Timestamp, pd.DataFrame], *,
    diagonal_reference: bool = False,
) -> qis.RiskModel:
    """Build QIS risk without replacing the chosen asset covariance.

    Args:
        model: Completed FM model or dated plain covariance matrices. Convert raw
            FL inputs with snapshots_from_factorlasso before calling this function.
        diagonal_reference: Explicitly include QIS's diagonal F/D reference for
            correlated residual results. Its factor decomposition describes that
            reference, while covariance-based risk uses the authoritative Sigma.

    Default correlated-residual export carries B, native coordinates, groups and
    full Sigma but omits QIS's F/D decomposition, which cannot represent full R.
    Residual histories, full R, cluster diagnostics and fit uncertainty remain FM
    data. Pure covariance input carries no factor-decomposition information.
    """
    if isinstance(model, dict) and model and all(isinstance(value, pd.DataFrame) for value in model.values()):
        if diagonal_reference:
            raise ValueError("covariance-only input has no factor or residual-policy metadata")
        return qis.RiskModel(covar={date: value.copy() for date, value in model.items()})

    if isinstance(model, FactorModelSnapshot):
        snapshots = {model.cutoff: model}
    elif isinstance(model, RollingFactorModel):
        snapshots = model.snapshots
    else:
        raise ValueError(f"unsupported or empty model: {type(model).__name__}; use snapshots_from_factorlasso for raw FL")
    first = next(iter(snapshots.values()))
    for snapshot in snapshots.values():
        if snapshot.factor_return_types != first.factor_return_types or snapshot.factor_groups != first.factor_groups:
            raise ValueError("QIS export requires consistent factor metadata across snapshots")
    residuals = {date: snapshot.residual_covariance for date, snapshot in snapshots.items()}
    has_correlated_residuals = any(
        np.any(matrix.to_numpy() != np.diag(np.diag(matrix.to_numpy())))
        for matrix in residuals.values()
    )
    include_decomposition = diagonal_reference or not has_correlated_residuals
    return qis.RiskModel(
        covar={date: snapshot.asset_covariance for date, snapshot in snapshots.items()},
        factor_loadings={date: snapshot.loadings for date, snapshot in snapshots.items()},
        factor_covar=({date: snapshot.factor_covariance for date, snapshot in snapshots.items()}
                      if include_decomposition else None),
        residual_vars=({
            date: pd.Series(np.diag(matrix), index=matrix.index)
            for date, matrix in residuals.items()
        } if include_decomposition else None),
        factor_return_types=first.factor_return_types,
        factor_groups=first.factor_groups,
    )


__all__ = [
    'build_risk_model',
]
