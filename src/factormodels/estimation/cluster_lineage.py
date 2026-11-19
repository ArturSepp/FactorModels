"""Prepare descriptive lineage statistics from completed native factor models.

Purpose: Supply lineage reports and reusable fingerprints for research sweeps.
Inputs: Completed FM results and explicit FL statistical matching settings.
Outputs: FL statistical reports or plain fingerprint fields, without fitted containers.
Units: Annual factor and raw residual variances retain their original convention.
Conditioning: Full-history lineage is descriptive; it is not a causal trading signal.
Ownership: FL owns fingerprinting and matching; FM owns fitted-result preparation.
"""
from dataclasses import asdict

from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, analyze_cluster_lineage
from factorlasso.diagnostics._lineage import RiskClusterReport, TaxonomyConfig, _snapshot_fingerprints

from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel

__all__ = ['prepare_cluster_lineage', 'prepare_cluster_fingerprints']


def _snapshot_inputs(snapshot: FactorModelSnapshot) -> CurrentFactorCovarData:
    """Prepare temporary statistical inputs; no FL fitted object leaves this module."""
    if not isinstance(snapshot, FactorModelSnapshot):
        raise TypeError('snapshot must be a completed FactorModelSnapshot')
    return CurrentFactorCovarData(
        x_covar=snapshot.factor_covariance, y_betas=snapshot.loadings,
        y_variances=snapshot.fit_statistics, clusters=snapshot.clusters,
        estimation_date=snapshot.cutoff)


def prepare_cluster_lineage(
    model: RollingFactorModel, *, overlap_metric: str = 'overlap',
    beta_metric: str = 'mahalanobis_sf', combine: str = 'gated',
    overlap_band: tuple[float, float] = (.15, .60), spread_vol_cut: float = .015,
    bridge_window: int = 6, bridge_decay: float = .5, w_overlap: float = .6,
    weighting: str = 'equal', taxonomy: TaxonomyConfig | None = None, method: str = 'mcf',
) -> RiskClusterReport:
    """Prepare FL's descriptive lineage report from native completed state.

    The statistical report contains tables and tracks, never fitted containers.
    Matching, labels and weighting retain FL's defaults unless explicitly supplied.
    Raw residual variances are used for descriptive fingerprints independently of
    the selected portfolio covariance policy, as in the original FL analysis.

    Args:
        model: Complete rolling factor-model history to describe.
        overlap_metric: FL membership overlap metric.
        beta_metric: FL factor-loading similarity metric.
        combine: Gated or blended matching rule.
        overlap_band: Lower and upper overlap thresholds.
        spread_vol_cut: Annual factor-spread volatility gate.
        bridge_window: Maximum unmatched-track bridge window in snapshots.
        bridge_decay: Similarity decay across an unmatched bridge.
        w_overlap: Membership weight in blended matching.
        weighting: Equal or inverse-volatility cluster fingerprints.
        taxonomy: Optional FL descriptive classification settings.
        method: FL lineage matcher, defaulting to minimum-cost flow.

    Returns:
        RiskClusterReport: Statistical tracks, classifications and label/membership panels.

    Raises:
        TypeError: If model is not a native rolling result.
    """
    if not isinstance(model, RollingFactorModel):
        raise TypeError('model must be a completed RollingFactorModel')
    inputs = RollingFactorCovarData({
        date: _snapshot_inputs(snapshot) for date, snapshot in model.snapshots.items()})
    return analyze_cluster_lineage(inputs, overlap_metric=overlap_metric,
        beta_metric=beta_metric, combine=combine, overlap_band=overlap_band,
        spread_vol_cut=spread_vol_cut, bridge_window=bridge_window,
        bridge_decay=bridge_decay, w_overlap=w_overlap, weighting=weighting,
        taxonomy=taxonomy, method=method)


def prepare_cluster_fingerprints(
    snapshot: FactorModelSnapshot, *, weighting: str = 'equal',
) -> tuple[dict, list[str]]:
    """Prepare owned plain fingerprint fields for statistical matching experiments.

    Args:
        snapshot: Completed result with fitted cluster memberships.
        weighting: Equal or inverse-volatility weighting, retaining FL semantics.

    Returns:
        tuple: Cluster-keyed plain fields and their ordered factor names. The fields
        contain members, beta, factor_var, idio_var, total_var, r2 and dominant.

    Raises:
        ValueError: If weighting is not an explicitly supported choice.
    """
    if weighting not in ('equal', 'inv_vol'):
        raise ValueError('weighting must be equal or inv_vol')
    fingerprints, factors = _snapshot_fingerprints(_snapshot_inputs(snapshot), weighting=weighting)
    return {label: asdict(value) for label, value in fingerprints.items()}, list(factors)
