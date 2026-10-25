"""
Report clustering of covar matrix.

Consumes completed FM risk and explicitly prepared per-date diagnostics.
Fitting and historical-alpha preparation belong to the caller.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250221

Purpose: Render completed covariance, cluster and residual diagnostics.
Inputs: Completed snapshots or RollingFactorModel plus plotting and output settings.
Outputs: Snapshot and rolling report functions produce QIS-based figures for the supplied results.
Units: Covariances are annual squared native-return coordinates; displayed volatility is annual decimal.
Conditioning: Use prepared linkage and selected residual risk without fitting or calling FL from reports.
Ownership: FM arranges method-specific views; QIS owns plotting primitives and estimation prepares all statistical inputs.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from qis.plots.utils import get_table_lines_for_group_data, set_suptitle
from matplotlib.colors import ListedColormap
from typing import List, Dict, Tuple, Mapping

from factormodels.analytics.models import (
    FactorModelSnapshot, RollingFactorModel, PreparedSnapshotDiagnostics,
)

__all__ = [
    'plot_covariance_snapshot',
    'plot_hcgl_covar_data',
    'run_rolling_covar_report',
]


def _snapshot_table(snapshot: FactorModelSnapshot,
                    diagnostics: PreparedSnapshotDiagnostics) -> pd.DataFrame:
    """Validate and retrieve explicitly prepared display inputs."""
    return diagnostics.for_snapshot(snapshot)


def plot_covariance_snapshot(snapshot: FactorModelSnapshot, *,
                             diagnostics: PreparedSnapshotDiagnostics,
                             **kwargs: object) -> List[plt.Figure]:
    """Render selected covariance using explicitly prepared historical alpha."""
    df = _snapshot_table(snapshot, diagnostics)
    required = {'r2', 'total_vol', 'resid_vol', 'stat_alpha'}
    if not required.issubset(df.columns):
        raise ValueError('covariance plot requires full prepared fit diagnostics')
    if snapshot.clusters_by_frequency is None:
        raise ValueError('covariance plot requires prepared cluster diagnostics')
    return plot_hcgl_covar_data(
        x_covar=snapshot.factor_covariance, y_covar=snapshot.asset_covariance,
        betas=snapshot.loadings, r2=df['r2'], total_vol=df['total_vol'],
        residual_vol=df['resid_vol'], alpha=df['stat_alpha'],
        clusters=snapshot.clusters_by_frequency, linkages=snapshot.linkages_by_frequency,
        cutoffs=snapshot.cutoffs_by_frequency, date=snapshot.cutoff, **kwargs,
    )


def plot_hcgl_covar_data(x_covar: pd.DataFrame,
                         y_covar: pd.DataFrame,
                         betas: pd.DataFrame,
                         r2: pd.Series,
                         total_vol: pd.Series,
                         residual_vol: pd.Series,
                         clusters: Dict[str, pd.Series],
                         linkages: Dict[str, np.ndarray],
                         cutoffs: Dict[str, float],
                         date: pd.Timestamp,
                         alpha: pd.Series = None,
                         figsize: Tuple[float, float] = (14, 10),
                         is_align_to_clusters_index: bool = True,
                         **kwargs: Any
                         ) -> List[plt.Figure]:
    """
    Plot covariance analysis for a single date.

    Returns:
        List of figures with factor/asset correlations, clusters, and betas.
    """
    figs = []

    # Prepare betas and stats
    betas = betas.where(np.abs(betas) > 1e-4, other=np.nan)
    if alpha is not None:
        df = pd.concat([r2.clip(0.0, None), total_vol, residual_vol, alpha], axis=1, sort=False)
    else:
        df = pd.concat([r2.clip(0.0, None), total_vol, residual_vol], axis=1, sort=False)

    agg_clusters, fig_clusters = qis.plot_clusters(
        clusters=clusters,
        linkages=linkages,
        cutoffs=cutoffs,
        figsize=figsize
    )

    if is_align_to_clusters_index:
        y_covar = y_covar.loc[agg_clusters.index, agg_clusters.index]

    # Plot factor correlations
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Factor correlations / vols")
    figs.append(fig)
    qis.plot_corr_matrix_from_covar(covar=x_covar,
                                    corr_format='{:.2f}',
                                    ax=ax,
                                    **kwargs)

    # Plot asset correlations
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Asset correlations / vols")
    figs.append(fig)
    qis.plot_corr_matrix_from_covar(covar=y_covar,
                                    corr_format='{:.1f}',
                                    ax=ax,
                                    **kwargs)

    # Add clusters figure
    set_suptitle(fig_clusters, title=f"{date.strftime('%d%b%Y')}: Clusters")
    figs.append(fig_clusters)

    # Plot betas and R²
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.0, hspace=0.0)
    axs = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Universe betas")
    figs.append(fig)

    hline_rows = get_table_lines_for_group_data(agg_clusters)
    if is_align_to_clusters_index:
        betas = betas.loc[agg_clusters.index, :]
        df = df.loc[agg_clusters.index, :]

    qis.plot_heatmap(df=betas, var_format='{:.2f}', hline_rows=hline_rows, ax=axs[0])

    # Create custom colormap
    colors = ['white']
    custom_cmap = ListedColormap(colors)
    qis.plot_heatmap(df=df, cmap=custom_cmap, var_format='{:.1%}',
                     hline_rows=hline_rows, ax=axs[1])

    return figs


def run_rolling_covar_report(
    model: RollingFactorModel, *, diagnostics: Mapping[pd.Timestamp, PreparedSnapshotDiagnostics],
    figsize: Tuple[float, float] = (14, 10), is_plot: bool = True,
    is_align_to_clusters_index: bool = True,
) -> Tuple[List[plt.Figure], Dict[str, pd.DataFrame]]:
    """Render completed rolling models and prepared dated diagnostics.

    The caller fits once, then explicitly prepares alpha with its chosen span and
    cadence settings. Exact date and asset alignment is checked before rendering.
    Returns figures and date-labelled table copies, without estimation or FL calls.
    """
    if not isinstance(model, RollingFactorModel):
        raise TypeError('model must be a completed RollingFactorModel')
    if set(diagnostics) != set(model.snapshots):
        raise ValueError('prepared diagnostics dates must match the rolling snapshots')
    tables = {date: _snapshot_table(snapshot, diagnostics[date])
              for date, snapshot in model.snapshots.items()}
    figs, dfs = [], {}
    for date, snapshot in model.snapshots.items():
        if is_plot:
            figs.extend(plot_covariance_snapshot(
                snapshot, diagnostics=diagnostics[date], figsize=figsize,
                is_align_to_clusters_index=is_align_to_clusters_index))
            plt.close('all')
        dfs[date.strftime('%d%b%Y')] = tables[date]
    return figs, dfs
