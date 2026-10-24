"""Completed in-memory factor models and dated covariance inputs.

FactorLasso containers retain fit diagnostics. These FM objects additionally
record the selected residual policy, native coordinates and cutoff, so downstream
users receive the covariance that was actually selected during construction.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, ResidualType, VarianceColumns
from qis.perfstats.config import ReturnTypes


def _return_types(factors: pd.Index, mapping: Mapping | None) -> dict:
    """Validate an exact factor coordinate map, defaulting to log returns."""
    result = dict(mapping) if mapping is not None else dict.fromkeys(factors, ReturnTypes.LOG)
    if set(result) != set(factors):
        raise ValueError("factor_return_types must match exactly the fitted factors")
    if any(value not in (ReturnTypes.LOG, ReturnTypes.DIFFERENCE) for value in result.values()):
        raise ValueError("factor_return_types supports LOG or DIFFERENCE")
    return result


@dataclass(frozen=True)
class DatedFactorCovariance:
    """Annual factor covariance with explicit observation/availability dates.

    Args:
        matrix: Square annual covariance in the factors' native return coordinates.
        observation_date: Latest observation used by the estimate.
        available_date: Date on which this estimate was available to the caller.
        factor_return_types: Exact factor coordinate map; omitted means LOG.

    The dates are caller-supplied provenance declarations, not independent proof
    of when upstream information was published.
    """

    matrix: pd.DataFrame
    observation_date: pd.Timestamp
    available_date: pd.Timestamp
    factor_return_types: dict | None = None

    def __post_init__(self) -> None:
        """Validate covariance labels, numerical shape and chronological provenance."""
        matrix = self.matrix
        if (not isinstance(matrix, pd.DataFrame) or matrix.empty
                or not matrix.index.is_unique or not matrix.columns.is_unique
                or not matrix.index.equals(matrix.columns)):
            raise ValueError("factor covariance must be nonempty, square and identically labelled")
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
            raise ValueError("factor covariance must be finite and symmetric")
        tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
        if np.linalg.eigvalsh(values).min() < -tolerance:
            raise ValueError("factor covariance must be positive semidefinite")
        observation = pd.Timestamp(self.observation_date)
        available = pd.Timestamp(self.available_date)
        if pd.isna(observation) or pd.isna(available) or observation > available:
            raise ValueError("factor covariance observation_date must be <= available_date")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "observation_date", observation)
        object.__setattr__(self, "available_date", available)
        object.__setattr__(self, "factor_return_types", _return_types(matrix.index, self.factor_return_types))

    def at_cutoff(self, cutoff: pd.Timestamp, factors: pd.Index, return_types: dict) -> pd.DataFrame:
        """Return an aligned copy only when provenance and native units match."""
        if self.available_date > cutoff or self.observation_date > cutoff:
            raise ValueError("supplied factor covariance is not available at cutoff")
        if set(self.matrix.index) != set(factors):
            raise ValueError("supplied factor covariance labels must match the factors")
        if self.factor_return_types != return_types:
            raise ValueError("supplied factor covariance return types disagree with the factor inputs")
        return self.matrix.reindex(index=factors, columns=factors).copy()


@dataclass(frozen=True)
class FactorModelSnapshot:
    """Completed factor decomposition and selected asset risk at one cutoff.

    Args:
        data: Full FactorLasso fitted decomposition, including residual histories.
        cutoff: Latest permitted information date.
        residual_type: Selected diagonal, empirical or cluster residual policy.
        residual_var_weight: Multiplier on the complete selected residual block.
        residual_corr_weight: Retention of nonorthogonal residual correlations.
        factor_return_types: Exact native factor coordinate map.
        factor_groups: Optional QIS factor group specifications.
        provenance: Input/covariance provenance recorded by the producer.

    The constructor owns a copy of fitted inputs. Pandas fields remain inspectable;
    consumers should treat them as result data and use a new result for edits.
    """

    data: CurrentFactorCovarData
    cutoff: pd.Timestamp
    residual_type: str = "orthogonal"
    residual_var_weight: float = 1.
    residual_corr_weight: float = 1.
    factor_return_types: dict | None = None
    factor_groups: dict | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate selected risk policy and retain independent fitted state."""
        cutoff = pd.Timestamp(self.cutoff)
        if pd.isna(cutoff):
            raise ValueError("cutoff must be a valid date")
        factors = self.data.x_covar
        betas = self.data.y_betas
        if (factors.empty or betas.empty or not factors.index.is_unique or not factors.columns.is_unique
                or not factors.index.equals(factors.columns)
                or not betas.columns.equals(factors.index)
                or not betas.index.is_unique
                or not self.data.y_variances.index.equals(betas.index)):
            raise ValueError("fitted covariance, loading and variance labels must align in the same order")
        factor_values = factors.to_numpy(dtype=float)
        if (not np.isfinite(factor_values).all()
                or not np.allclose(factor_values, factor_values.T, rtol=1e-10, atol=1e-12)
                or not np.isfinite(betas.to_numpy(dtype=float)).all()):
            raise ValueError("fitted factor covariance and loadings must be finite; covariance must be symmetric")
        tolerance = 1e-10 * max(1., float(np.max(np.abs(factor_values))))
        if np.linalg.eigvalsh(factor_values).min() < -tolerance:
            raise ValueError("fitted factor covariance must be positive semidefinite")
        residual_values = self.data.y_variances[VarianceColumns.RESIDUAL_VARS.value].to_numpy(dtype=float)
        if not np.isfinite(residual_values).all() or np.any(residual_values < 0):
            raise ValueError("fitted residual variances must be finite and nonnegative")
        if self.data.estimation_date is None or pd.Timestamp(self.data.estimation_date) != cutoff:
            raise ValueError("fitted data estimation_date must equal cutoff")
        if self.data.residuals is not None and not self.data.residuals.empty:
            if self.data.residuals.index.max() > cutoff:
                raise ValueError("fitted residual history extends beyond cutoff")
        kind = ResidualType(self.residual_type).value
        if not np.isfinite(self.residual_var_weight) or self.residual_var_weight < 0:
            raise ValueError("residual_var_weight must be finite and nonnegative")
        if not np.isfinite(self.residual_corr_weight) or not 0 <= self.residual_corr_weight <= 1:
            raise ValueError("residual_corr_weight must be finite and in [0, 1]")
        if kind == "orthogonal" and self.residual_corr_weight != 1.:
            raise ValueError("residual_corr_weight applies only to nonorthogonal residuals")
        object.__setattr__(self, "data", deepcopy(self.data))
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "residual_type", kind)
        object.__setattr__(self, "factor_return_types", _return_types(self.data.x_covar.index, self.factor_return_types))
        object.__setattr__(self, "factor_groups", deepcopy(self.factor_groups))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        # Validate readiness now; a completed model must not fail on first retrieval.
        self.data.get_residual_covar(
            residual_var_weight=self.residual_var_weight,
            residual_type=kind, residual_corr_weight=self.residual_corr_weight,
        )

    @property
    def loadings(self) -> pd.DataFrame:
        """Return a copy of the fitted assets-by-factors loadings in native units.

        Labels and ordering are retained from the completed fit at ``cutoff``.
        Retrieval performs no refitting or coordinate conversion.
        """
        return self.data.y_betas.copy()

    @property
    def factor_covariance(self) -> pd.DataFrame:
        """Return a copy of annual factor covariance in the fitted native coordinates.

        Both axes retain the fitted factor order. Retrieval performs no refitting
        or change to the fit's annualisation or ``cutoff``.
        """
        return self.data.x_covar.copy()

    @property
    def asset_covariance(self) -> pd.DataFrame:
        """Return annual B F B' plus the selected complete residual covariance."""
        return self.data.get_y_covar(
            residual_var_weight=self.residual_var_weight,
            residual_type=self.residual_type,
            residual_corr_weight=self.residual_corr_weight,
        )

    @property
    def residual_covariance(self) -> pd.DataFrame:
        """Return the selected annual residual block, including its variance weight."""
        return self.data.get_residual_covar(
            residual_var_weight=self.residual_var_weight,
            residual_type=self.residual_type,
            residual_corr_weight=self.residual_corr_weight,
        )

    def to_risk_model(self, *, diagonal_reference: bool = False) -> qis.RiskModel:
        """Export authoritative total risk with supported QIS factor metadata.

        Correlated residuals retain B and total covariance but omit QIS's diagonal
        F/D decomposition unless explicitly requesting a diagonal reference view.
        """
        from factormodels.estimation.risk import build_risk_model

        return build_risk_model(self, diagonal_reference=diagonal_reference)


@dataclass(frozen=True)
class RollingFactorModel:
    """Completed dated factor models with inspectable FL and covariance views."""

    snapshots: dict[pd.Timestamp, FactorModelSnapshot]

    def __post_init__(self) -> None:
        """Require a nonempty, uniquely dated mapping with consistent snapshot dates."""
        if not self.snapshots:
            raise ValueError("rolling model must contain at least one snapshot")
        snapshots = {pd.Timestamp(date): value for date, value in self.snapshots.items()}
        if len(snapshots) != len(self.snapshots):
            raise ValueError("rolling model has duplicate normalized dates")
        for date, snapshot in snapshots.items():
            if not isinstance(snapshot, FactorModelSnapshot) or date != snapshot.cutoff:
                raise ValueError("rolling keys must equal each FactorModelSnapshot cutoff")
        object.__setattr__(self, "snapshots", dict(sorted(snapshots.items())))

    @property
    def data(self) -> RollingFactorCovarData:
        """Expose complete fitted decompositions as a FactorLasso rolling container."""
        return RollingFactorCovarData({date: snapshot.data for date, snapshot in self.snapshots.items()})

    @property
    def covariances(self) -> dict[pd.Timestamp, pd.DataFrame]:
        """Return the selected annual asset covariance at every fitted date."""
        return {date: snapshot.asset_covariance for date, snapshot in self.snapshots.items()}

    def to_risk_model(self, *, diagonal_reference: bool = False) -> qis.RiskModel:
        """Export all snapshots without replacing their selected residual policies."""
        from factormodels.estimation.risk import build_risk_model

        return build_risk_model(self, diagonal_reference=diagonal_reference)
