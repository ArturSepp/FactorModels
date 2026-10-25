"""Fit each native return cadence through FactorLasso.

Purpose: Fit each native return cadence through FactorLasso.
Inputs: Dated asset returns, factor prices and the configured FL estimator.
Outputs: Unannualised loadings, fit statistics, residual histories and clustering metadata.
Units: Residuals retain native return units; variance is squared return per observation.
Conditioning: Resolve cadence maps before fitting; restore temporary penalty and HAC overrides on failure.
Ownership: FM orchestrates cadence inputs; FL owns regression and QIS owns return conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel

__all__: list[str] = []

def _model_for_frequency(lasso_model: LassoModel, freq: str) -> LassoModel:
    """Resolve regression and clustering spans for one return cadence.

    Args:
        lasso_model: FactorLasso configuration, optionally carrying cadence maps.
        freq: Asset-return frequency code used to look up each configured map.

    Returns:
        A model copy with span overrides when maps are present; otherwise the
        supplied model itself. Spans count observations at the requested cadence.

    Raises:
        KeyError: If a configured regression or clustering span map lacks freq.
    """
    overrides = {}
    if lasso_model.span_freq_dict is not None:
        if freq not in lasso_model.span_freq_dict:
            raise KeyError(f"no span for freq={freq} in lasso_model.span_freq_dict")
        overrides['span'] = lasso_model.span_freq_dict[freq]
    cluster_span_map = getattr(
        lasso_model, 'cluster_correlation_span_freq_dict', None
    )
    if cluster_span_map is not None:
        if freq not in cluster_span_map:
            raise KeyError(
                f"no cluster correlation span for freq={freq} in "
                "lasso_model.cluster_correlation_span_freq_dict"
            )
        overrides['cluster_correlation_span'] = cluster_span_map[freq]
    prior_hac_map = getattr(lasso_model, 'expert_prior_hac_lags_freq_dict', None)
    if (prior_hac_map is not None
            and getattr(lasso_model, 'expert_prior_bound_n_std', None) is not None):
        if freq not in prior_hac_map:
            raise KeyError(f"no expert prior HAC bandwidth for freq={freq}")
        overrides['expert_prior_hac_lags'] = prior_hac_map[freq]
    return lasso_model.copy(kwargs=overrides) if overrides else lasso_model


@dataclass(frozen=True)
class _FrequencyFitResult:
    """Unannualized components from one asset-return cadence.

    Attributes:
        betas: Asset-by-factor loadings from the fitted model.
        ewma_variances: Per-asset total variation reported by the fit.
        residual_variances: Per-asset residual variation reported by the fit.
        alphas: Per-asset fitted intercepts.
        r2: Dimensionless fit R-squared diagnostics.
        clusters: Optional per-asset fitted cluster assignments.
        linkage: Optional SciPy-style tree for this cadence.
        cutoff: Optional dendrogram cut distance.
        residuals: Supplied asset returns minus fitted factor returns times betas;
            the fitted intercept is not subtracted.
        derived_signs: Optional asset-by-factor sign requirements; NaN is unconstrained.
    """

    betas: pd.DataFrame
    ewma_variances: pd.Series
    residual_variances: pd.Series
    alphas: pd.Series
    r2: pd.Series
    clusters: Optional[pd.Series]
    linkage: Optional[np.ndarray]
    cutoff: Optional[float]
    residuals: pd.DataFrame
    derived_signs: Optional[pd.DataFrame]


def _fit_lasso_frequency(
        *,
        freq: str,
        asset_returns: pd.DataFrame,
        risk_factor_prices: pd.DataFrame,
        lasso_model: LassoModel,
        verbose: bool,
        precomputed_clusters: Optional[Dict[str, pd.Series]] = None,
        precomputed_linkages: Optional[Dict[str, np.ndarray]] = None,
        precomputed_cutoffs: Optional[Dict[str, float]] = None,
        reg_lambda: Optional[float] = None,
        factor_return_types: Optional[dict] = None,
) -> _FrequencyFitResult:
    """Fit one return cadence and collect unannualized model components.

    Factor prices are aligned with historical forward-filling to asset dates,
    then converted to log returns. Cadence maps override regression/clustering
    spans for the fit. The original LassoModel receives the fitted state; a
    combined multi-cadence result must be read from the returned containers.

    Args:
        freq: Frequency code for this asset-return bucket.
        asset_returns: Date-by-asset log returns at the bucket's observation cadence.
        risk_factor_prices: Date-by-factor prices covering the observation history.
        lasso_model: Configured model to fit in place.
        verbose: Whether to print solver diagnostics.
        precomputed_clusters: Optional cadence-to-assignment map. A matching key
            supplies external memberships without replacing the configured model type.
        precomputed_linkages: Corresponding linkage map, required when memberships
            are supplied for this cadence.
        precomputed_cutoffs: Corresponding cut-distance map, required when memberships
            are supplied for this cadence.
        reg_lambda: Optional fixed penalty for this cadence. The model's scalar
            configuration is restored after fitting, including on solver failure.

    Returns:
        Unannualized fit components. Residual time series exclude only the fitted
        factor contribution, not the regression intercept.

    Raises:
        ValueError: If supplied cluster assignments are missing for any fitted asset.
        KeyError: If required cadence spans, linkages or cutoffs are absent.
    """
    factor_prices = risk_factor_prices.reindex(index=asset_returns.index, method='ffill').ffill()
    factor_returns = qis.to_factor_returns(
        prices=factor_prices,
        return_types=factor_return_types,
        is_first_zero=False,
        drop_first=False,
        freq=None,
    )

    frequency_model = _model_for_frequency(lasso_model=lasso_model, freq=freq)
    span = frequency_model.span
    cluster_correlation_span = getattr(
        frequency_model, 'cluster_correlation_span', None
    )

    use_precomputed = precomputed_clusters is not None and freq in precomputed_clusters
    external_clusters = None
    if use_precomputed:
        external_clusters = precomputed_clusters[freq].reindex(asset_returns.columns)
        if external_clusters.isna().any():
            missing = external_clusters[external_clusters.isna()].index.tolist()
            raise ValueError(
                f"precomputed_clusters[{freq!r}] is missing assignments "
                f"for {len(missing)} assets: {missing}"
            )

    # Preserve the public function's fitted-state contract: callers receive their original
    # LassoModel back with the final cadence's fit attached. A fresh model copy would be a separate
    # numerical and behavioural change, even though clustering schedules use that pattern safely.
    fit_model = lasso_model
    cluster_span_kwargs = {}
    if hasattr(fit_model, 'cluster_correlation_span'):
        cluster_span_kwargs['cluster_correlation_span'] = cluster_correlation_span
    original_lambda = fit_model.reg_lambda
    original_prior_hac_lags = getattr(fit_model, 'expert_prior_hac_lags', None)
    if original_prior_hac_lags is not None:
        fit_model.expert_prior_hac_lags = frequency_model.expert_prior_hac_lags
    if reg_lambda is not None:
        fit_model.reg_lambda = reg_lambda
    try:
        fit_model.fit(
            x=factor_returns,
            y=asset_returns,
            verbose=verbose,
            span=span,
            external_clusters=external_clusters,
            external_linkage=precomputed_linkages[freq] if use_precomputed else None,
            external_cutoff=precomputed_cutoffs[freq] if use_precomputed else None,
            **cluster_span_kwargs,
        )
    finally:
        fit_model.reg_lambda = original_lambda
        if original_prior_hac_lags is not None:
            fit_model.expert_prior_hac_lags = original_prior_hac_lags

    # Match the regression's missing-predictor treatment when constructing the
    # factor contribution. A late-starting factor must not propagate NaN through
    # every response, including those with zero exposure to that factor. Keep
    # wholly unavailable factor rows and missing asset observations unavailable.
    factor_contribution = factor_returns.fillna(0.0) @ fit_model.estimated_betas.T
    factor_contribution.loc[factor_returns.isna().all(axis=1), :] = np.nan

    estimation_result = fit_model.estimation_result_
    linkage = precomputed_linkages[freq] if use_precomputed else fit_model.linkage
    cutoff = precomputed_cutoffs[freq] if use_precomputed else fit_model.cutoff
    return _FrequencyFitResult(
        betas=fit_model.estimated_betas,
        ewma_variances=pd.Series(estimation_result.ss_total, index=asset_returns.columns),
        residual_variances=pd.Series(estimation_result.ss_res, index=asset_returns.columns),
        alphas=pd.Series(estimation_result.alpha, index=asset_returns.columns),
        r2=pd.Series(estimation_result.r2, index=asset_returns.columns),
        clusters=fit_model.clusters,
        linkage=linkage,
        cutoff=cutoff,
        residuals=asset_returns - factor_contribution,
        derived_signs=fit_model.derived_signs_,
    )
