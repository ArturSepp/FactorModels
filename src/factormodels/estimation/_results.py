"""Materialize FL estimates into FM-owned completed annual moments.

FL owns residual covariance assembly and historical residual-alpha estimation.
This boundary copies its exact outputs, native cadence/scaling evidence and
observation/availability dates. Inspection of the resulting FM objects needs
neither an FL container nor estimation. No annualisation is inferred or repeated.

Purpose: Materialise completed FM contracts at the FactorLasso estimation boundary.
Inputs: FL fitted containers or supplied moments, cutoffs, residual policy and coordinate metadata.
Outputs: Snapshots, historical-alpha estimates and prepared diagnostics contain no retained FL container.
Units: F, R and Sigma retain annual native coordinates; alpha and residual-history scaling remain explicit.
Conditioning: Copy exact selected and reference covariance views once; never infer or repeat annualisation.
Ownership: FL assembles covariance and estimates alpha; FM materialises results for analytics and reports.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import (
    CurrentFactorCovarData, RollingFactorCovarData, ResidualCorrelationData, ResidualType,
    get_clusters_by_freq, get_linkages_by_freq, get_cutoffs_by_freq,
)

from factormodels.analytics.models import (
    FactorModelSnapshot, RollingFactorModel, PreparedSnapshotDiagnostics,
)

__all__ = ['snapshots_from_factorlasso', 'snapshot_from_moments',
           'estimate_historical_alpha', 'prepare_snapshot_diagnostics']


def _materialize(
    data: CurrentFactorCovarData, *, cutoff: pd.Timestamp, residual_type: str,
    residual_var_weight: float = 1., residual_corr_weight: float = 1.,
    factor_return_types: dict | None = None, factor_groups: dict | None = None,
    provenance: dict | None = None,
) -> FactorModelSnapshot:
    """Copy exact upstream outputs once, retaining no upstream object."""
    cutoff = pd.Timestamp(cutoff)
    residual_type = ResidualType(residual_type).value
    if data.estimation_date is not None and pd.Timestamp(data.estimation_date) != cutoff:
        raise ValueError('fitted data estimation_date must equal cutoff')
    # Preserve filtered-universe FL targets: never rebuild the raw fitted object
    # for risk assembly (dataclasses.replace would lose FL's init=False targets).
    options = dict(residual_type=residual_type, residual_var_weight=residual_var_weight,
                   residual_corr_weight=residual_corr_weight)
    residual = data.get_residual_covar(**options)
    covariance = data.get_y_covar(**options)
    reference = data.get_y_covar(residual_type='orthogonal', residual_var_weight=1.)
    diagnostics = None
    if {'r2', 'insample_alpha'}.issubset(data.y_variances.columns):
        # FL supplies display order and raw volatilities. No alpha span is selected
        # at materialization: its no-history path supplies the span-free columns.
        diagnostics = replace(data, residuals=None).get_snapshot().drop(columns='stat_alpha')
    corr = data.residual_correlation
    corr_metadata = None if corr is None else dict(
        residual_returns=corr.residual_returns, asset_metadata=corr.asset_metadata,
        frequency=corr.frequency, span=corr.span, observation_date=corr.observation_date,
        estimation_date=corr.estimation_date, schema_version=corr.schema_version,
    )
    return FactorModelSnapshot(
        loadings=data.y_betas, factor_covariance=data.x_covar,
        fit_statistics=data.y_variances, fit_diagnostics=diagnostics,
        residual_covariance=residual, asset_covariance=covariance,
        raw_diagonal_reference_covariance=reference, cutoff=cutoff,
        residual_type=residual_type, residual_var_weight=residual_var_weight,
        residual_corr_weight=residual_corr_weight, factor_return_types=factor_return_types,
        factor_groups=factor_groups, provenance=provenance,
        residuals=data.residuals, residual_metadata=data.residual_metadata,
        residual_correlation=None if corr is None else corr.correlation,
        residual_correlation_metadata=corr_metadata,
        clusters=data.clusters, linkages=data.linkages, cutoffs=data.cutoffs,
        derived_signs=data.derived_signs,
        clusters_by_frequency=None if data.clusters is None else get_clusters_by_freq(data.clusters),
        linkages_by_frequency=None if data.linkages is None else get_linkages_by_freq(data.linkages),
        cutoffs_by_frequency=None if data.cutoffs is None else get_cutoffs_by_freq(data.cutoffs),
    )


def snapshots_from_factorlasso(
    raw: CurrentFactorCovarData | RollingFactorCovarData | Mapping[pd.Timestamp, CurrentFactorCovarData], *,
    residual_type: str, residual_var_weight: float = 1., residual_corr_weight: float = 1.,
    factor_return_types: dict | None = None, factor_groups: dict | None = None,
    provenance: dict | None = None,
) -> RollingFactorModel:
    """Select a residual policy and materialize raw FL data into dated FM results.

    A single FL result requires an estimation date; dated mappings may supply a
    missing date. Existing dates must agree with keys. Filtered FL block targets
    are consumed by FL before copying. No input container is mutated or retained.
    """
    if isinstance(raw, CurrentFactorCovarData):
        if raw.estimation_date is None:
            raise ValueError('single raw result requires estimation_date or a dated mapping')
        raw = {raw.estimation_date: raw}
    elif isinstance(raw, RollingFactorCovarData):
        raw = raw.data
    if not isinstance(raw, Mapping) or not raw or not all(
            isinstance(value, CurrentFactorCovarData) for value in raw.values()):
        raise ValueError('raw must contain dated FactorLasso fitted results')
    dates = [pd.Timestamp(date) for date in raw]
    if len(set(dates)) != len(dates):
        raise ValueError('raw FactorLasso data has duplicate normalized dates')
    return RollingFactorModel({date: _materialize(
        value, cutoff=date, residual_type=residual_type, residual_var_weight=residual_var_weight,
        residual_corr_weight=residual_corr_weight, factor_return_types=factor_return_types,
        factor_groups=factor_groups, provenance=provenance,
    ) for date, value in zip(dates, raw.values())})


def snapshot_from_moments(
    betas: pd.DataFrame, factor_covariance: pd.DataFrame, residual_variances: pd.Series, *,
    date: pd.Timestamp, residual_correlation: ResidualCorrelationData | None = None,
    residual_type: str = 'orthogonal', residual_var_weight: float = 1.,
    residual_corr_weight: float = 1., factor_return_types: dict | None = None,
    factor_groups: dict | None = None, provenance: dict | None = None,
) -> FactorModelSnapshot:
    """Prepare supplied annual moments using FL's canonical risk assembly.

    Minimal inputs do not invent histories, fit R-squared or in-sample alpha.
    Optional prepared FL residual dependence is consumed here and stored as plain
    fields with separate observation and estimation dates. Select its policy
    explicitly. Cluster policies require the fuller raw-FL adapter instead.
    """
    data = CurrentFactorCovarData(
        x_covar=factor_covariance, y_betas=betas,
        y_variances=residual_variances.rename('residual_var').to_frame(),
        estimation_date=pd.Timestamp(date), residual_correlation=residual_correlation,
    )
    return _materialize(data, cutoff=date, residual_type=residual_type,
                        residual_var_weight=residual_var_weight, residual_corr_weight=residual_corr_weight,
                        factor_return_types=factor_return_types, factor_groups=factor_groups, provenance=provenance)


def _alpha_inputs(snapshot: FactorModelSnapshot) -> CurrentFactorCovarData:
    """Reconstruct only the plain fields required by FL alpha/display methods."""
    if not isinstance(snapshot, FactorModelSnapshot):
        raise TypeError('snapshot must be a completed FactorModelSnapshot')
    return CurrentFactorCovarData(
        x_covar=snapshot.factor_covariance, y_betas=snapshot.loadings,
        y_variances=snapshot.fit_statistics, estimation_date=snapshot.cutoff,
        residuals=snapshot.residuals, residual_metadata=snapshot.residual_metadata,
    )


def estimate_historical_alpha(
    snapshot: FactorModelSnapshot, *, alpha_span: int | dict[str, int] = 120,
    asset_frequencies: str | pd.Series | None = None, default_freq: str = 'ME',
) -> pd.Series:
    """Delegate historical alpha to FL with unchanged spans, gaps and scaling.

    Scalar and cadence-map paths preserve FL semantics, including its native
    frequency metadata fallback. Without history, preserve FL get_snapshot's
    in-sample-alpha fallback; if neither source exists, raise ValueError.
    """
    raw = _alpha_inputs(snapshot)
    if raw.residuals is not None:
        return raw.estimate_alpha(alpha_span=alpha_span, asset_frequencies=asset_frequencies,
                                  default_freq=default_freq)
    if {'r2', 'insample_alpha'}.issubset(raw.y_variances.columns):
        return raw.get_snapshot(alpha_span=alpha_span, asset_frequencies=asset_frequencies,
                                default_freq=default_freq)['stat_alpha']
    raise ValueError('historical alpha unavailable: no residual history or in-sample fit diagnostics')


def prepare_snapshot_diagnostics(
    snapshot: FactorModelSnapshot, *, alpha_span: int | dict[str, int] = 120,
    asset_frequencies: str | pd.Series | None = None, default_freq: str = 'ME',
) -> PreparedSnapshotDiagnostics:
    """Prepare explicit alpha and selected-risk display inputs for pure consumers.

    The table preserves FL's display column order when full fit diagnostics exist.
    Minimal moments expose their supplied statistics with unavailable stat_alpha
    as NaN. No fabricated R-squared, history, or in-sample alpha is added.
    """
    raw = _alpha_inputs(snapshot)
    settings = dict(alpha_span=alpha_span, asset_frequencies=asset_frequencies, default_freq=default_freq)
    full = {'r2', 'insample_alpha'}.issubset(raw.y_variances.columns)
    available = raw.residuals is not None or full
    source = 'residual_history' if raw.residuals is not None else ('insample_alpha' if full else 'unavailable')
    if full:
        table = raw.get_snapshot(**settings)
    else:
        table = pd.concat([raw.y_betas, raw.y_variances, raw.get_model_vols()], axis=1)
        table['stat_alpha'] = estimate_historical_alpha(snapshot, **settings) if available else np.nan
    table['total_vol'] = np.sqrt(np.diag(snapshot.asset_covariance))
    table['resid_vol'] = np.sqrt(np.diag(snapshot.residual_covariance))
    return PreparedSnapshotDiagnostics(table, cutoff=snapshot.cutoff, alpha_settings=settings,
                                       alpha_source=source, alpha_available=available)
