"""Prepare factor-derived cluster inputs and scores for downstream consumers.

The outputs are ordinary dated Series mappings or dates-by-assets score panels.
Consumers can use these completed inputs without importing FactorLasso or
retaining a fitted model. Signal formulas and portfolio optimisation stay with
the caller; FactorLasso owns the statistical stability-pooling calculation.

Purpose: Prepare dated cluster assignments and stability-pooled score panels.
Inputs: Completed rolling models, supplied signal values and declared pooling settings.
Outputs: extract_rolling_clusters and prepare_cluster_scores return ordinary labelled pandas outputs.
Units: Cluster identities are categorical; scores retain the supplied signal convention.
Conditioning: Use dated completed assignments and the configured stability pooling, without inventing signal formulas.
Ownership: FL owns statistical pooling; FM prepares its inputs and applications own trading decisions.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import qis
from factorlasso import (
    StabilityPoolingType,
    score_with_stability_pooled_clusters,
)

from factormodels.analytics.models import RollingFactorModel

__all__ = [
    'extract_rolling_clusters',
    'prepare_cluster_scores',
]


def extract_rolling_clusters(
    model: RollingFactorModel,
    assets: Sequence[str] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Extract completed cluster assignments from a rolling fitted model.

    Args:
        model: Completed FM rolling factor model. Raw FL inputs need the explicit adapter.
        assets: Optional ordered asset universe. Missing assignments are dropped.

    Returns:
        Estimation dates mapped to independent asset-indexed cluster Series.
        Empty dates are omitted. Frequency-prefixed labels remain unchanged;
        duplicate asset labels retain their last assignment.
    """
    if not isinstance(model, RollingFactorModel):
        raise TypeError('model must be a completed RollingFactorModel')
    rolling_clusters = {}
    for date, current_data in model.snapshots.items():
        clusters = current_data.clusters
        if clusters is None or len(clusters) == 0:
            continue
        clusters = clusters[~clusters.index.duplicated(keep="last")]
        if assets is not None:
            clusters = clusters.reindex(assets).dropna()
        if len(clusters) > 0:
            rolling_clusters[date] = clusters.copy()
    return rolling_clusters


def prepare_cluster_scores(
    raw_signal: pd.DataFrame,
    rolling_clusters: dict[pd.Timestamp, pd.Series],
    *,
    min_cluster_size: int = 3,
    pooling_type: StabilityPoolingType | str = StabilityPoolingType.NONE,
    stability_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare completed cluster scores, with optional FL stability pooling.

    Args:
        raw_signal: Dates-by-assets raw signals supplied by the caller.
        rolling_clusters: Dated assignments, for example from
            :func:`extract_rolling_clusters`. Each signal date uses the latest
            assignment available on or before it.
        min_cluster_size: Clusters at or below this size use global moments.
        pooling_type: FL variance-pooling mode or its string value. ``"none"``
            retains unpooled scoring and ignores stability weights.
        stability_weights: Causal dates-by-assets weights in [0, 1], required
            for pooled modes. Weights are resolved at each assignment date,
            so later weights do not modify earlier assignments.

    Returns:
        A completed score panel with the input date and asset order. Pass it
        directly as supplied alpha scores to a downstream consumer. Dates before
        the first assignment score zero. With no assignments and no pooling,
        QIS's clipped, population-standard-deviation global scorer is retained;
        FL's pooled modes retain their own global fallback convention.

    Raises:
        ValueError: If a pooled mode lacks weights or has invalid weights.
    """
    pooling_type = StabilityPoolingType(pooling_type)
    if pooling_type == StabilityPoolingType.NONE:
        if not rolling_clusters:
            return qis.df_to_cross_sectional_score(df=raw_signal)
        stability_weights = None
    return score_with_stability_pooled_clusters(
        raw_signal=raw_signal,
        rolling_clusters=rolling_clusters,
        stability_weights=stability_weights,
        min_cluster_size=min_cluster_size,
        pooling_type=pooling_type,
    )
