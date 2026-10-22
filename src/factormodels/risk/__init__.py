"""Export selected FM risk to the existing QIS RiskModel contract."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData

from factormodels.model import FactorModelSnapshot, RollingFactorModel


def build_risk_model(
    model: FactorModelSnapshot | RollingFactorModel | RollingFactorCovarData | dict, *,
    residual_type: str | None = None,
    residual_var_weight: float = 1.,
    residual_corr_weight: float = 1.,
    factor_return_types: dict | None = None,
    factor_groups: dict | None = None,
    diagonal_reference: bool = False,
) -> qis.RiskModel:
    """Build QIS risk without replacing the chosen asset covariance.

    Args:
        model: Completed FM model, dated plain covariance matrices, or raw FL
            decompositions with an explicitly selected residual_type.
        residual_type: Required for raw FL data, which has no selected policy.
            FM results already carry the policy and reject overrides.
        residual_var_weight: Selected raw-FL residual multiplier.
        residual_corr_weight: Selected raw-FL residual correlation retention.
        factor_return_types: Native coordinates for raw FL data.
        factor_groups: Factor family metadata for raw FL data.
        diagonal_reference: Explicitly include QIS's diagonal F/D reference for
            correlated residual results. Its factor decomposition describes that
            reference, while covariance-based risk uses the authoritative Sigma.

    Default correlated-residual export carries B, native coordinates, groups and
    full Sigma but omits QIS's F/D decomposition, which cannot represent full R.
    Residual histories, full R, cluster diagnostics and fit uncertainty remain FM
    data. Pure covariance input carries no factor-decomposition information.
    """
    if isinstance(model, dict) and model and all(isinstance(value, pd.DataFrame) for value in model.values()):
        if (residual_type is not None or factor_return_types is not None or factor_groups is not None
                or residual_var_weight != 1. or residual_corr_weight != 1. or diagonal_reference):
            raise ValueError("covariance-only input has no factor or residual-policy metadata")
        return qis.RiskModel(covar={date: value.copy() for date, value in model.items()})

    if isinstance(model, FactorModelSnapshot):
        snapshots = {model.cutoff: model}
    elif isinstance(model, RollingFactorModel):
        snapshots = model.snapshots
    else:
        raw = model.data if isinstance(model, RollingFactorCovarData) else model
        if not isinstance(raw, dict) or not raw or not all(
                isinstance(value, CurrentFactorCovarData) for value in raw.values()):
            raise ValueError(f"unsupported or empty model: {type(model).__name__}")
        if residual_type is None:
            raise ValueError("raw FactorLasso data requires explicit residual_type")
        dates = [pd.Timestamp(date) for date in raw]
        if len(set(dates)) != len(dates):
            raise ValueError("raw FactorLasso data has duplicate normalized dates")
        snapshots = {
            date: FactorModelSnapshot(
                data=(value if value.estimation_date is not None
                      else replace(value, estimation_date=date)),
                cutoff=date, residual_type=residual_type,
                residual_var_weight=residual_var_weight,
                residual_corr_weight=residual_corr_weight,
                factor_return_types=factor_return_types, factor_groups=factor_groups,
            ) for date, value in zip(dates, raw.values())
        }

    if isinstance(model, (FactorModelSnapshot, RollingFactorModel)) and (
            residual_type is not None or residual_var_weight != 1. or residual_corr_weight != 1.
            or factor_return_types is not None or factor_groups is not None):
        raise ValueError("completed FM models already carry policy and factor metadata")
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
        factor_loadings={date: snapshot.data.y_betas.copy() for date, snapshot in snapshots.items()},
        factor_covar=({date: snapshot.data.x_covar.copy() for date, snapshot in snapshots.items()}
                      if include_decomposition else None),
        residual_vars=({
            date: pd.Series(np.diag(matrix), index=matrix.index)
            for date, matrix in residuals.items()
        } if include_decomposition else None),
        factor_return_types=first.factor_return_types,
        factor_groups=first.factor_groups,
    )
