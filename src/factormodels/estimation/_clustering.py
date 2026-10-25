"""Prepare current and rolling factor-assisted cluster schedules.

Purpose: Prepare current and rolling factor-assisted cluster schedules.
Inputs: Native return panels, factor references, cadence configuration and estimation dates.
Outputs: FL cluster paths and per-cadence assignments, restricted linkages and cutoffs.
Units: Assignments are dimensionless; tree heights retain the configured correlation distance.
Conditioning: Preserve warmup eligibility, current endpoints and rolling schedule order without refitting trees.
Ownership: FL discovers and smooths clusters; FM schedules calls and restricts reporting leaves.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import List
import numpy as np
import pandas as pd
import qis
from factorlasso import ClusterSmootherType, LassoModel, RollingClusterData, compute_rolling_smoothed_clusters
from factormodels.estimation._frequency import _model_for_frequency

__all__: list[str] = []

def _validate_recluster_frequency(recluster_freq: str, rebalancing_freq: str) -> None:
    """Check that the reclustering frequency generates fewer calendar anchors.

    The check compares counts over a fixed 2000-2029 calendar interval. It checks
    relative cadence, not whether every recluster date is a covariance date.

    Args:
        recluster_freq: Pandas frequency string for reclustering anchors.
        rebalancing_freq: Pandas frequency string for covariance output dates.

    Raises:
        ValueError: If either frequency is invalid or reclustering generates at
            least as many anchors as covariance rebalancing.
    """
    sample_start = pd.Timestamp('2000-01-01')
    sample_end = pd.Timestamp('2029-12-31')
    try:
        recluster_count = len(pd.date_range(sample_start, sample_end, freq=recluster_freq))
        rebalancing_count = len(pd.date_range(sample_start, sample_end, freq=rebalancing_freq))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid recluster_freq={recluster_freq!r} or "
            f"rebalancing_freq={rebalancing_freq!r}"
        ) from exc
    if recluster_count >= rebalancing_count:
        raise ValueError(
            f"recluster_freq={recluster_freq!r} must be coarser than "
            f"rebalancing_freq={rebalancing_freq!r}"
        )


# FL's get_linkage_array exports a whole tree, not an induced leaf subset.
# This adapter retains merge heights without reclustering.
def _restrict_linkage(linkage: np.ndarray, labels: pd.Index,
                      assets: pd.Index) -> np.ndarray:
    """Remove reference leaves from a discovered clustering tree.

    The adapter retains discovered merge heights and the requested asset order;
    it does not recluster the retained assets.

    Args:
        linkage: SciPy-style linkage array for the complete discovered tree.
        labels: Leaf labels in the order used by linkage.
        assets: Ordered subset of leaf labels to retain.

    Returns:
        Four-column linkage array over retained assets; zero or one retained
        leaf gives no merge rows.

    Raises:
        ValueError: If any requested asset label is absent from labels.
    """
    positions = labels.get_indexer(assets)
    if np.any(positions < 0):
        raise ValueError('clustering linkage is missing fitted asset labels')
    nodes = {int(old): (new, 1) for new, old in enumerate(positions)}
    rows = []
    for step, (left, right, height, _) in enumerate(linkage):
        lhs, rhs = nodes.get(int(left)), nodes.get(int(right))
        parent = len(labels) + step
        if lhs is not None and rhs is not None:
            count = lhs[1] + rhs[1]
            nodes[parent] = (len(assets) + len(rows), count)
            rows.append((lhs[0], rhs[0], height, count))
        elif lhs is not None or rhs is not None:
            nodes[parent] = lhs if lhs is not None else rhs
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def _factor_cluster_path(risk_factor_prices: pd.DataFrame, returns: pd.DataFrame,
                         schedule: List[pd.Timestamp], model: LassoModel, factor_return_types: dict | None = None) -> RollingClusterData:
    """Build factor-assisted cluster paths and retain asset-only reporting trees.

    Factor prices are forward-filled to the return index and converted to log
    returns. FactorLasso discovers clusters on assets plus renamed factor
    references. References are then removed from assignments and linkages.
    Missing asset assignments receive distinct groups; warm-up-eligible asset
    leaves determine each reported tree.

    Args:
        risk_factor_prices: Date-by-factor prices for clustering references.
        returns: One cadence's date-by-asset log returns.
        schedule: Estimation dates passed to FactorLasso's cluster-path builder.
        model: Configuration with spans resolved for this return cadence.

    Returns:
        FactorLasso cluster-path container with asset assignments and restricted
        linkages. Its cutoff records remain those of the discovered reference tree.

    Raises:
        ValueError: If asset/factor columns are not unique, asset names collide
            with reserved factor-anchor labels, or retained linkage labels are missing.
    """
    if not returns.columns.is_unique or not risk_factor_prices.columns.is_unique:
        raise ValueError('factor clustering requires unique asset and factor columns')
    prices = risk_factor_prices.reindex(index=returns.index, method='ffill').ffill()
    factors = qis.to_factor_returns(
        prices, return_types=factor_return_types, is_first_zero=False, drop_first=False, freq=None,
    )
    factors.columns = pd.Index([f'__factor_anchor__:{i}' for i in range(len(factors.columns))])
    if not returns.columns.intersection(factors.columns).empty:
        raise ValueError('asset names collide with reserved __factor_anchor__: labels')
    panel = pd.concat([returns, factors], axis=1, sort=True)
    path = compute_rolling_smoothed_clusters(y=panel, estimation_dates=schedule, lasso_model=model)
    for date in schedule:
        discovered = path.clusters[date]
        assignments = discovered.reindex(returns.columns).copy()
        # A common-mode transform may omit not-yet-eligible assets from discovery. Give each
        # missing response its own solver group; FactorLasso still zeroes/drops it at warmup.
        next_group = int(discovered.max()) + 1
        for asset in assignments.index[assignments.isna()]:
            assignments.loc[asset] = next_group
            next_group += 1
        active = returns.columns
        if model.warmup_period is not None:
            active = active[returns.loc[:date].notna().sum() >= model.warmup_period]
        path.linkages[date] = _restrict_linkage(path.linkages[date], discovered.index, active)
        path.clusters[date] = assignments.astype(int)
    return path


def _current_cluster_inputs(
    risk_factor_prices: pd.DataFrame, asset_returns_dict: dict[str, pd.DataFrame],
    lasso_model: LassoModel, estimation_date: pd.Timestamp | None,
    rebalancing_freq: str, include_factors_in_clustering: bool,
    use_factor_references: Callable[[str], bool], factor_return_types: dict | None,
    precomputed_clusters: dict[str, pd.Series] | None,
    precomputed_linkages: dict[str, np.ndarray] | None,
    precomputed_cutoffs: dict[str, float] | None,
) -> tuple[pd.Timestamp | None, dict[str, pd.Series] | None,
           dict[str, np.ndarray] | None, dict[str, float] | None]:
    """Prepare endpoint memberships unless the caller supplies all partitions."""
    smoother_type = ClusterSmootherType(lasso_model.cluster_smoother_type)
    if (smoother_type != ClusterSmootherType.NONE or include_factors_in_clustering
            ) and precomputed_clusters is None:
        estimation_date = estimation_date or max(
            returns.index[-1] for returns in asset_returns_dict.values()
        )
        precomputed_clusters = {}
        precomputed_linkages = {}
        precomputed_cutoffs = {}
        for freq, returns in asset_returns_dict.items():
            if (smoother_type == ClusterSmootherType.NONE
                    and not use_factor_references(freq)):
                continue
            fit_model = _model_for_frequency(lasso_model, freq)
            start_position = min((fit_model.warmup_period or 1) - 1, len(returns.index) - 1)
            start_date = returns.index[start_position]
            schedule = qis.generate_dates_schedule(
                time_period=qis.TimePeriod(start_date, estimation_date),
                freq=rebalancing_freq,
                include_start_date=False,
                include_end_date=True,
            )
            final_date = pd.Timestamp(estimation_date)
            if smoother_type == ClusterSmootherType.NONE:
                schedule = [final_date]
            if final_date not in schedule:
                # Not covered, and currently unreachable: the schedule above is generated with
                # include_end_date=True and estimation_date as the end, so qis always returns
                # it whether or not the date falls on the rebalancing grid. Kept as a guard in
                # case that flag or its semantics change -- the lookups below index the
                # smoother output by final_date and would raise if it were ever absent.
                schedule = sorted([*schedule, final_date])  # pragma: no cover
            rolling_clusters = (
                _factor_cluster_path(risk_factor_prices, returns, schedule, fit_model,
                                     factor_return_types)
                if use_factor_references(freq) else compute_rolling_smoothed_clusters(
                    y=returns, estimation_dates=schedule, lasso_model=fit_model,
                )
            )
            precomputed_clusters[freq] = rolling_clusters.clusters[final_date]
            precomputed_linkages[freq] = rolling_clusters.linkages[final_date]
            precomputed_cutoffs[freq] = rolling_clusters.cutoffs[final_date]

    return estimation_date, precomputed_clusters, precomputed_linkages, precomputed_cutoffs


def _rolling_cluster_paths(
    risk_factor_prices: pd.DataFrame, asset_returns_dict: dict[str, pd.DataFrame],
    lasso_model: LassoModel, rebalancing_schedule: Sequence[pd.Timestamp],
    effective_rebalancing_freq: str, include_factors_in_clustering: bool,
    use_factor_references: Callable[[str], bool], factor_return_types: dict | None,
) -> dict[str, RollingClusterData] | None:
    """Prepare cadence paths on exactly the covariance output schedule."""
    smoother_type = ClusterSmootherType(lasso_model.cluster_smoother_type)
    rolling_clusters_by_freq = None
    if smoother_type != ClusterSmootherType.NONE or include_factors_in_clustering:
        if lasso_model.recluster_freq is not None:
            _validate_recluster_frequency(
                recluster_freq=str(lasso_model.recluster_freq),
                rebalancing_freq=effective_rebalancing_freq,
            )
        rolling_clusters_by_freq = {
            freq: (
                _factor_cluster_path(
                    risk_factor_prices, returns, rebalancing_schedule,
                    _model_for_frequency(lasso_model, freq),
                    factor_return_types,
                ) if use_factor_references(freq) else compute_rolling_smoothed_clusters(
                    y=returns, estimation_dates=rebalancing_schedule,
                    lasso_model=_model_for_frequency(lasso_model, freq),
                )
            )
            for freq, returns in asset_returns_dict.items()
            if smoother_type != ClusterSmootherType.NONE or use_factor_references(freq)
        }

    return rolling_clusters_by_freq
