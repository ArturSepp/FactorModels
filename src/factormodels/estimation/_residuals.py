"""Prepare selected residual correlation and reuse unchanged rolling observations.

Purpose: Prepare selected residual correlation and reuse unchanged rolling observations.
Inputs: Annual-scaled residual histories, native-cadence metadata and marginal residual variances.
Outputs: Prepared FL residual correlation and snapshots sharing an unchanged correlation observation.
Units: Correlations are dimensionless; metadata records scaling from native return observations.
Conditioning: Estimate positive-risk assets only, keep unfitted assets independent and preserve observation-date reuse.
Ownership: FL owns residual estimation; FM supplies policy, cadence, scaling and container orchestration.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, ResidualCorrelationData

__all__: list[str] = []

def _validate_residual_type(residual_type: str) -> None:
    """Validate the four supported residual covariance policies."""
    choices = ('orthogonal', 'empirical', 'exposure_cluster', 'residual_cluster')
    if residual_type not in choices:
        raise ValueError(f"residual_type must be one of {choices}")


def _reuse_residual_correlation(
    current: CurrentFactorCovarData, previous_residual: ResidualCorrelationData | None,
) -> tuple[CurrentFactorCovarData, ResidualCorrelationData]:
    """Reuse a correlation observation without freezing the current marginal variances."""
    prepared = current.residual_correlation
    if (previous_residual is not None
            and prepared.observation_date == previous_residual.observation_date
            and prepared.frequency == previous_residual.frequency
            and prepared.span == previous_residual.span
            and prepared.correlation.index.equals(previous_residual.correlation.index)
            and prepared.asset_metadata.equals(previous_residual.asset_metadata)):
        prepared = previous_residual
        current = replace(current, residual_correlation=prepared)
    return current, prepared


def _prepare_residual_correlation(
    residuals: pd.DataFrame, metadata: pd.DataFrame, last_residual_vars: pd.Series,
    estimation_date: pd.Timestamp, residual_covar_freq: str | None,
    residual_covar_span: float | None,
) -> ResidualCorrelationData:
    """Delegate common-grid estimation and restore independent zero-risk output assets."""
    from factorlasso import estimate_residual_correlation

    annualisation = (qis.get_annualisation_conversion_factor(residual_covar_freq, 'YE')
                     if residual_covar_freq is not None else None)
    # A reindexed output universe can contain assets with no fitted history. Their
    # zero marginal residual variance makes correlation undefined but contributes no
    # covariance; estimate the supported block and keep those assets independent.
    positive_risk = last_residual_vars.index[last_residual_vars.gt(0)]
    if positive_risk.empty:
        raise ValueError('Empirical residual correlation needs positive residual risk')
    has_zero_risk = len(positive_risk) != len(residuals.columns)
    fitted_correlation = estimate_residual_correlation(
        residuals=residuals[positive_risk] if has_zero_risk else residuals,
        metadata=metadata.loc[positive_risk] if has_zero_risk else metadata,
        estimation_date=estimation_date,
        frequency=residual_covar_freq, span=residual_covar_span,
        periods_per_year=annualisation,
    )
    if has_zero_risk:
        names = residuals.columns
        correlation = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        correlation.loc[positive_risk, positive_risk] = fitted_correlation.correlation
        fitted_correlation = replace(
            fitted_correlation, correlation=correlation,
            residual_returns=fitted_correlation.residual_returns.reindex(
                columns=names, fill_value=0.0),
            asset_metadata=metadata,
        )
    return fitted_correlation
