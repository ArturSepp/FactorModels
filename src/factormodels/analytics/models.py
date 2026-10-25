"""Completed in-memory factor models and dated covariance inputs.

FM owns plain completed fields in annual native coordinates. Estimation prepares
all covariance views before construction; inspection performs no fitting or FL
assembly. Mutable result fields are copied on entry and on public retrieval.

Purpose: Store completed factor-model snapshots and dated covariance inputs.
Inputs: Already materialised loadings, F, selected R/Sigma, fit tables and provenance.
Outputs: FactorModelSnapshot and RollingFactorModel expose owned, copy-protected completed moments.
Units: Sigma = B F B.T + selected R in annual native factor coordinates; metadata records residual scaling.
Conditioning: Cutoff, label order and selected residual policy are explicit; access never recomputes FL state.
Ownership: FM owns completed contracts; estimation materialises them and QIS consumes risk interchange views.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from qis import ReturnTypes


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



@dataclass(frozen=True, init=False)
class FactorModelSnapshot:
    """Owned completed moments, fit evidence and point-in-time provenance.

    All matrices use the exact fitted asset/factor order. ``fit_statistics`` is
    the raw fit table; ``fit_diagnostics`` is an optional span-free display table.
    Selected R/Sigma already include residual policy weights. The raw diagonal
    reference uses unit residual variance weight and is distinct from QIS's
    selected-R diagonal decomposition. Use estimation adapters to prepare these
    views from FL or supplied moments; this constructor does no covariance assembly.

    Pandas/NumPy fields and nested metadata are owned copies. Public retrieval
    returns independent copies. No FactorLasso containers are retained.
    """

    _fields: dict
    cutoff: pd.Timestamp
    residual_type: str
    residual_var_weight: float
    residual_corr_weight: float

    def __init__(
        self, loadings: pd.DataFrame, factor_covariance: pd.DataFrame,
        residual_covariance: pd.DataFrame, asset_covariance: pd.DataFrame,
        raw_diagonal_reference_covariance: pd.DataFrame, fit_statistics: pd.DataFrame,
        *, cutoff: pd.Timestamp, residual_type: str = "orthogonal",
        residual_var_weight: float = 1., residual_corr_weight: float = 1.,
        factor_return_types: dict | None = None, factor_groups: dict | None = None,
        provenance: dict | None = None, fit_diagnostics: pd.DataFrame | None = None,
        residuals: pd.DataFrame | None = None, residual_metadata: pd.DataFrame | None = None,
        residual_correlation: pd.DataFrame | None = None,
        residual_correlation_metadata: dict | None = None,
        clusters: pd.Series | None = None, linkages: pd.DataFrame | None = None,
        cutoffs: pd.Series | None = None, derived_signs: pd.DataFrame | None = None,
        clusters_by_frequency: dict | None = None, linkages_by_frequency: dict | None = None,
        cutoffs_by_frequency: dict | None = None,
    ) -> None:
        """Validate completed labels, dimensions, dates and selected policy."""
        date = pd.Timestamp(cutoff)
        if pd.isna(date):
            raise ValueError("cutoff must be a valid date")
        b, f = loadings, factor_covariance
        if (b.empty or f.empty or not b.index.is_unique or not b.columns.is_unique
                or not f.index.equals(f.columns) or not b.columns.equals(f.index)
                or not fit_statistics.index.equals(b.index)):
            raise ValueError("fitted covariance, loading and variance labels must align in the same order")
        _validate_covariance(f, b.columns, "factor covariance")
        if not np.isfinite(b.to_numpy(dtype=float)).all():
            raise ValueError("fitted loadings must be finite")
        d = fit_statistics['residual_var'].to_numpy(dtype=float)
        if not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("fitted residual variances must be finite and nonnegative")
        for name, matrix in (("residual covariance", residual_covariance),
                             ("asset covariance", asset_covariance),
                             ("raw diagonal reference covariance", raw_diagonal_reference_covariance)):
            _validate_covariance(matrix, b.index, name, check_psd=False)
        kind = str(residual_type)
        if kind not in ('orthogonal', 'empirical', 'exposure_cluster', 'residual_cluster'):
            raise ValueError("unknown residual_type")
        if not np.isfinite(residual_var_weight) or residual_var_weight < 0:
            raise ValueError("residual_var_weight must be finite and nonnegative")
        if not np.isfinite(residual_corr_weight) or not 0 <= residual_corr_weight <= 1:
            raise ValueError("residual_corr_weight must be finite and in [0, 1]")
        if kind == 'orthogonal' and residual_corr_weight != 1.:
            raise ValueError("residual_corr_weight applies only to nonorthogonal residuals")
        for name, table in (("fit_diagnostics", fit_diagnostics), ("residual_metadata", residual_metadata),
                            ("derived_signs", derived_signs)):
            if table is not None and not table.index.equals(b.index):
                raise ValueError(f"{name} must match the fitted asset order")
        if derived_signs is not None and not derived_signs.columns.equals(b.columns):
            raise ValueError("derived_signs must match the fitted factor order")
        if residuals is not None:
            _validate_history(residuals, b.index, date, 'fitted residual history')
        if (residual_correlation is None) != (residual_correlation_metadata is None):
            raise ValueError("residual correlation requires its provenance metadata")
        if residual_correlation is not None:
            _validate_covariance(residual_correlation, b.index, "residual correlation", psd_tolerance=1e-8)
            if not np.allclose(np.diag(residual_correlation), 1., rtol=0, atol=1e-12):
                raise ValueError("residual correlation must have unit diagonal")
            info = residual_correlation_metadata
            observation, estimation = pd.Timestamp(info['observation_date']), pd.Timestamp(info['estimation_date'])
            if pd.isna(observation) or pd.isna(estimation) or not observation <= estimation <= date:
                raise ValueError("residual correlation observation/estimation dates must be available at cutoff")
            _validate_history(info['residual_returns'], b.index, observation, 'common residual history')
            if not info['asset_metadata'].index.equals(b.index):
                raise ValueError("residual correlation asset metadata must align")
        values = dict(loadings=b, factor_covariance=f, residual_covariance=residual_covariance,
                      asset_covariance=asset_covariance,
                      raw_diagonal_reference_covariance=raw_diagonal_reference_covariance,
                      fit_statistics=fit_statistics, fit_diagnostics=fit_diagnostics,
                      residuals=residuals, residual_metadata=residual_metadata,
                      residual_correlation=residual_correlation,
                      residual_correlation_metadata=residual_correlation_metadata,
                      clusters=clusters, linkages=linkages, cutoffs=cutoffs, derived_signs=derived_signs,
                      clusters_by_frequency=clusters_by_frequency, linkages_by_frequency=linkages_by_frequency,
                      cutoffs_by_frequency=cutoffs_by_frequency,
                      factor_return_types=_return_types(b.columns, factor_return_types),
                      factor_groups=factor_groups, provenance=provenance or {})
        _require_plain(values)
        owned = deepcopy(values)
        # One owned object suffices for identical immutable internal views.
        if (asset_covariance.equals(raw_diagonal_reference_covariance)
                and asset_covariance.to_numpy().tobytes() == raw_diagonal_reference_covariance.to_numpy().tobytes()
                and asset_covariance.index.names == raw_diagonal_reference_covariance.index.names
                and asset_covariance.columns.names == raw_diagonal_reference_covariance.columns.names
                and not asset_covariance.attrs and not raw_diagonal_reference_covariance.attrs):
            owned['raw_diagonal_reference_covariance'] = owned['asset_covariance']
        object.__setattr__(self, '_fields', owned)
        for name, value in dict(cutoff=date, residual_type=kind,
                                residual_var_weight=residual_var_weight,
                                residual_corr_weight=residual_corr_weight).items():
            object.__setattr__(self, name, value)

    @property
    def loadings(self) -> pd.DataFrame:
        """Return an independent copy of completed loadings."""
        return deepcopy(self._fields['loadings'])

    @property
    def factor_covariance(self) -> pd.DataFrame:
        """Return an independent copy of completed factor covariance."""
        return deepcopy(self._fields['factor_covariance'])

    @property
    def asset_covariance(self) -> pd.DataFrame:
        """Return an independent copy of completed asset covariance."""
        return deepcopy(self._fields['asset_covariance'])

    @property
    def residual_covariance(self) -> pd.DataFrame:
        """Return an independent copy of completed residual covariance."""
        return deepcopy(self._fields['residual_covariance'])

    @property
    def raw_diagonal_reference_covariance(self) -> pd.DataFrame:
        """Return an independent copy of completed raw diagonal reference covariance."""
        return deepcopy(self._fields['raw_diagonal_reference_covariance'])

    @property
    def fit_statistics(self) -> pd.DataFrame:
        """Return an independent copy of completed fit statistics."""
        return deepcopy(self._fields['fit_statistics'])

    @property
    def fit_diagnostics(self) -> pd.DataFrame | None:
        """Return an independent copy of completed fit diagnostics."""
        return deepcopy(self._fields['fit_diagnostics'])

    @property
    def residuals(self) -> pd.DataFrame | None:
        """Return an independent copy of completed residuals."""
        return deepcopy(self._fields['residuals'])

    @property
    def residual_metadata(self) -> pd.DataFrame | None:
        """Return an independent copy of completed residual metadata."""
        return deepcopy(self._fields['residual_metadata'])

    @property
    def residual_correlation(self) -> pd.DataFrame | None:
        """Return an independent copy of completed residual correlation."""
        return deepcopy(self._fields['residual_correlation'])

    @property
    def residual_correlation_metadata(self) -> dict | None:
        """Return an independent copy of completed residual correlation metadata."""
        return deepcopy(self._fields['residual_correlation_metadata'])

    @property
    def clusters(self) -> pd.Series | None:
        """Return an independent copy of completed clusters."""
        return deepcopy(self._fields['clusters'])

    @property
    def linkages(self) -> pd.DataFrame | None:
        """Return an independent copy of completed linkages."""
        return deepcopy(self._fields['linkages'])

    @property
    def cutoffs(self) -> pd.Series | None:
        """Return an independent copy of completed cutoffs."""
        return deepcopy(self._fields['cutoffs'])

    @property
    def derived_signs(self) -> pd.DataFrame | None:
        """Return an independent copy of completed derived signs."""
        return deepcopy(self._fields['derived_signs'])

    @property
    def clusters_by_frequency(self) -> dict | None:
        """Return an independent copy of completed clusters by frequency."""
        return deepcopy(self._fields['clusters_by_frequency'])

    @property
    def linkages_by_frequency(self) -> dict | None:
        """Return an independent copy of completed linkages by frequency."""
        return deepcopy(self._fields['linkages_by_frequency'])

    @property
    def cutoffs_by_frequency(self) -> dict | None:
        """Return an independent copy of completed cutoffs by frequency."""
        return deepcopy(self._fields['cutoffs_by_frequency'])

    @property
    def factor_return_types(self) -> dict:
        """Return an independent copy of completed factor return types."""
        return deepcopy(self._fields['factor_return_types'])

    @property
    def factor_groups(self) -> dict | None:
        """Return an independent copy of completed factor groups."""
        return deepcopy(self._fields['factor_groups'])

    @property
    def provenance(self) -> dict:
        """Return an independent copy of completed provenance."""
        return deepcopy(self._fields['provenance'])

    def to_risk_model(self, *, diagonal_reference: bool = False) -> qis.RiskModel:
        """Export authoritative total risk and supported QIS metadata."""
        from factormodels.analytics.risk import build_risk_model
        return build_risk_model(self, diagonal_reference=diagonal_reference)

    def with_provenance(self, provenance: Mapping) -> FactorModelSnapshot:
        """Return the same completed moments with independently owned caller provenance.

        This records currency, units or source declarations without changing risk
        policy or recomputing covariance. Internal completed arrays remain shared;
        every public read still returns a copy.
        """
        _require_plain(provenance)
        updated = object.__new__(type(self))
        for name in ('cutoff', 'residual_type', 'residual_var_weight', 'residual_corr_weight'):
            object.__setattr__(updated, name, getattr(self, name))
        object.__setattr__(updated, '_fields', {**self._fields, 'provenance': deepcopy(dict(provenance))})
        return updated


def _require_plain(value: object) -> None:
    """Reject embedded FL objects even inside caller-supplied metadata."""
    if type(value).__module__.split('.')[0] == 'factorlasso':
        raise TypeError("completed results cannot contain FactorLasso objects")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_plain(key)
            _require_plain(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _require_plain(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _require_plain(getattr(value, item.name))
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        _require_plain(value.attrs)
        objects = value.loc[:, value.dtypes == object].to_numpy().flat if isinstance(value, pd.DataFrame) else (
            value.to_numpy() if value.dtype == object else ())
        for item in objects:
            _require_plain(item)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            _require_plain(item)


def _validate_covariance(matrix: pd.DataFrame, axis: pd.Index, name: str, *,
                         check_psd: bool = True, psd_tolerance: float | None = None) -> None:
    """Check a supplied completed matrix without assembling or repairing it."""
    if not isinstance(matrix, pd.DataFrame) or not matrix.index.equals(axis) or not matrix.columns.equals(axis):
        raise ValueError(f"{name} labels must align in the same order")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be finite and symmetric")
    tolerance = psd_tolerance if psd_tolerance is not None else 1e-10 * max(1., float(np.max(np.abs(values))))
    if check_psd and np.linalg.eigvalsh(values).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")


def _validate_history(history: pd.DataFrame, assets: pd.Index, cutoff: pd.Timestamp, name: str) -> None:
    """Validate history axes and availability without altering missing observations."""
    if (not history.columns.equals(assets) or not isinstance(history.index, pd.DatetimeIndex)
            or history.index.hasnans or not history.index.is_unique):
        raise ValueError(f"{name} labels must align and dates must be unique")
    if not history.empty and history.index.max() > cutoff:
        raise ValueError(f"{name} extends beyond cutoff")


@dataclass(frozen=True, init=False)
class RollingFactorModel:
    """Completed FM snapshots; map retrieval cannot mutate the rolling result."""

    _snapshots: dict[pd.Timestamp, FactorModelSnapshot]

    def __init__(self, snapshots: dict[pd.Timestamp, FactorModelSnapshot]) -> None:
        """Require nonempty, uniquely dated snapshots with matching cutoffs."""
        if not snapshots:
            raise ValueError("rolling model must contain at least one snapshot")
        normalized = {pd.Timestamp(date): value for date, value in snapshots.items()}
        if len(normalized) != len(snapshots):
            raise ValueError("rolling model has duplicate normalized dates")
        for date, snapshot in normalized.items():
            if not isinstance(snapshot, FactorModelSnapshot) or date != snapshot.cutoff:
                raise ValueError("rolling keys must equal each FactorModelSnapshot cutoff")
        object.__setattr__(self, '_snapshots', dict(sorted(normalized.items())))

    @property
    def snapshots(self) -> dict[pd.Timestamp, FactorModelSnapshot]:
        """Return a new date map of immutable completed snapshots."""
        return dict(self._snapshots)

    @property
    def covariances(self) -> dict[pd.Timestamp, pd.DataFrame]:
        """Return the selected annual covariance at every fitted date."""
        return {date: snapshot.asset_covariance for date, snapshot in self._snapshots.items()}

    def to_risk_model(self, *, diagonal_reference: bool = False) -> qis.RiskModel:
        """Export selected rolling risk without reassembly."""
        from factormodels.analytics.risk import build_risk_model
        return build_risk_model(self, diagonal_reference=diagonal_reference)


@dataclass(frozen=True, init=False)
class PreparedSnapshotDiagnostics:
    """Explicit dated display table with alpha settings and availability.

    Alpha is historical residual attribution in producer-declared annual units,
    not a tactical forecast. Missing history falls back to in-sample alpha when
    available; minimal moments retain missing statistical evidence explicitly.
    """

    _table: pd.DataFrame
    _alpha_settings: dict
    cutoff: pd.Timestamp
    alpha_source: str
    alpha_available: bool

    def __init__(self, table: pd.DataFrame, *, cutoff: pd.Timestamp,
                 alpha_settings: dict, alpha_source: str, alpha_available: bool) -> None:
        """Own a dated labelled table and its declared preparation settings."""
        if not table.index.is_unique or pd.isna(pd.Timestamp(cutoff)):
            raise ValueError("prepared diagnostics require unique assets and a valid cutoff")
        _require_plain(alpha_settings)
        object.__setattr__(self, '_table', table.copy(deep=True))
        object.__setattr__(self, '_alpha_settings', deepcopy(alpha_settings))
        object.__setattr__(self, 'cutoff', pd.Timestamp(cutoff))
        object.__setattr__(self, 'alpha_source', alpha_source)
        object.__setattr__(self, 'alpha_available', alpha_available)

    @property
    def table(self) -> pd.DataFrame:
        """Return a copy of the prepared table."""
        return self._table.copy(deep=True)

    @property
    def alpha_settings(self) -> dict:
        """Return independently editable preparation settings."""
        return deepcopy(self._alpha_settings)

    def for_snapshot(self, snapshot: FactorModelSnapshot) -> pd.DataFrame:
        """Require matching cutoff and exact asset order before consumption."""
        if self.cutoff != snapshot.cutoff or not self._table.index.equals(snapshot.loadings.index):
            raise ValueError("prepared diagnostics must match snapshot cutoff and asset order")
        return self.table


__all__ = ['DatedFactorCovariance', 'FactorModelSnapshot', 'RollingFactorModel',
           'PreparedSnapshotDiagnostics']
