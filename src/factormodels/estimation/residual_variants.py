"""Explicit residual-policy preparation from native completed results.

Purpose: Prepare requested research/stress risk variants without refitting.
Inputs: FM snapshots, a selected residual policy and explicit variance/correlation weights.
Outputs: Completed snapshots or rolling models with selected annual R and Sigma.
Units: Existing annual moments, native residual histories and coordinate maps are preserved.
Conditioning: Use retained full-universe targets; causal burn-in is an explicit replay option.
Ownership: FL assembles risk from temporary inputs; FM owns the returned plain result state.
"""
from __future__ import annotations

from copy import deepcopy

from factorlasso import CurrentFactorCovarData, ResidualCorrelationData, ResidualType

from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel

__all__ = ["prepare_residual_variant"]


def _prepare_snapshot(
    snapshot: FactorModelSnapshot, kind: str, variance_weight: float, correlation_weight: float,
    *, burn_in: bool,
) -> FactorModelSnapshot:
    """Assemble through FL while retaining every original fit/evidence field."""
    fields = deepcopy(snapshot._fields)
    metadata = fields["residual_correlation_metadata"]
    correlation = None
    if metadata is not None:
        metadata = deepcopy(metadata)
        schema = metadata.pop("schema_version")
        correlation = ResidualCorrelationData(correlation=fields["residual_correlation"], **metadata)
        if correlation.schema_version != schema:
            raise ValueError("unsupported residual correlation schema")
    selected = kind
    if kind != "orthogonal" and correlation is None:
        if not burn_in:
            raise ValueError("selected residual correlation is unavailable outside declared burn-in")
        selected, correlation_weight = "orthogonal", 1.
    targets = fields["residual_cluster_targets"] or {}
    if (selected in ("exposure_cluster", "residual_cluster")
            and variance_weight > 0 and correlation_weight > 0 and selected not in targets):
        raise ValueError("requested residual cluster target was not retained from the original fit")
    raw = CurrentFactorCovarData(
        x_covar=fields["factor_covariance"], y_betas=fields["loadings"],
        y_variances=fields["fit_statistics"], estimation_date=snapshot.cutoff,
        residual_correlation=correlation, clusters=fields["clusters"],
    )
    # FL's public assembly consumes this plain cache; recomputing it on selected
    # assets would change the original-universe cluster means and partitions.
    object.__setattr__(raw, "_residual_cluster_targets", targets)
    options = dict(residual_type=selected, residual_var_weight=variance_weight,
                   residual_corr_weight=correlation_weight)
    fields["residual_covariance"] = raw.get_residual_covar(**options)
    fields["asset_covariance"] = raw.get_y_covar(**options)
    fields["provenance"].update(residual_policy_target=kind, residual_correlation_burn_in=selected != kind)
    return FactorModelSnapshot(**fields, cutoff=snapshot.cutoff, **options)


def prepare_residual_variant(
    model: FactorModelSnapshot | RollingFactorModel, *, residual_type: str,
    residual_var_weight: float = 1., residual_corr_weight: float = 1.,
    allow_orthogonal_burn_in: bool = False,
) -> FactorModelSnapshot | RollingFactorModel:
    """Prepare explicitly selected residual risk using retained native fit evidence.

    Args:
        model: Completed snapshot or rolling history. No fitting is performed.
        residual_type: Orthogonal, empirical, exposure_cluster or residual_cluster.
        residual_var_weight: Nonnegative multiplier on the annual residual block.
        residual_corr_weight: Retained correlation fraction; orthogonal requires one.
        allow_orthogonal_burn_in: Explicit historical replay declaration. Missing
            correlation may select orthogonal risk only before the first prepared
            correlation. A later gap fails, including after selection or reload.

    Returns:
        The same result kind with new selected covariance and policy provenance.
        Raw diagonal reference, fit statistics and diagnostic meanings are unchanged.

    Raises:
        ValueError: If evidence, a requested target, dates or policy are invalid.
        TypeError: If model is not a completed native FM result.
    """
    kind = ResidualType(residual_type).value
    # Validate even when a declared burn-in would otherwise replace the policy.
    import math

    if not math.isfinite(residual_var_weight) or residual_var_weight < 0:
        raise ValueError("residual_var_weight must be finite and nonnegative")
    if not math.isfinite(residual_corr_weight) or not 0 <= residual_corr_weight <= 1:
        raise ValueError("residual_corr_weight must be in [0, 1]")
    if kind == "orthogonal" and residual_corr_weight != 1.:
        raise ValueError("orthogonal residual_corr_weight must equal one")
    if isinstance(model, FactorModelSnapshot):
        return _prepare_snapshot(model, kind, residual_var_weight, residual_corr_weight,
                                 burn_in=allow_orthogonal_burn_in)
    if not isinstance(model, RollingFactorModel):
        raise TypeError("model must be FactorModelSnapshot or RollingFactorModel")
    prepared_before = False
    results = {}
    for date, snapshot in model.snapshots.items():
        results[date] = _prepare_snapshot(
            snapshot, kind, residual_var_weight, residual_corr_weight,
            burn_in=allow_orthogonal_burn_in and not prepared_before,
        )
        prepared_before |= snapshot.residual_correlation is not None
    return RollingFactorModel(results)
