"""Compose compatible orthogonal fitted blocks through the canonical FL assembly.

Purpose: Combine separately fitted asset blocks with their shared factor covariance.
Inputs: Native snapshots, explicit asset order and optional caller-supplied loadings/labels.
Outputs: One completed orthogonal snapshot with all cross-block factor risk retained.
Units: Annual native-coordinate covariance; raw fit and residual scaling are preserved.
Conditioning: Cutoff, factor axis, coordinates and residual weights must agree exactly.
Ownership: Applications choose blocks and cash overrides; FL assembles risk and FM owns the result.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import pandas as pd
from factorlasso import CurrentFactorCovarData

from factormodels.analytics.models import FactorModelSnapshot
from factormodels.analytics._selection import selected_assets
from factormodels.estimation._results import _materialize

__all__ = ["compose_snapshots"]


def _check_blocks(blocks: list[FactorModelSnapshot]) -> pd.Index:
    """Require disjoint compatible fitted blocks without silently aligning factors."""
    if not blocks or not all(isinstance(block, FactorModelSnapshot) for block in blocks):
        raise ValueError("snapshots must contain completed native blocks")
    first = blocks[0]
    for block in blocks:
        if block.cutoff != first.cutoff:
            raise ValueError("snapshot cutoffs must agree")
        if block.residual_type != "orthogonal" or block.residual_correlation is not None:
            raise ValueError("composition supports orthogonal blocks without cross-block residual dependence only")
        if block.residual_var_weight != first.residual_var_weight:
            raise ValueError("residual variance weights must agree")
        if (not block.factor_covariance.equals(first.factor_covariance)
                or not block.loadings.columns.equals(first.loadings.columns)
                or block.factor_return_types != first.factor_return_types):
            raise ValueError("blocks must share exact factor order, covariance and coordinates")
        if block.factor_groups != first.factor_groups:
            raise ValueError("factor groups must agree")
        for key in ("reference_ccy", "asset_return_type", "factor_covariance_units"):
            if block.provenance.get(key) != first.provenance.get(key):
                raise ValueError(f"conflicting or missing block declaration: {key}")
        metadata = block.provenance
        dates = [metadata[key] for key in (
            "factor_observation_date", "covariance_observation_date", "covariance_available_date",
            "clustering_available_date") if metadata.get(key) is not None]
        dates.extend(metadata.get("asset_observation_dates", {}).values())
        if any(pd.isna(pd.Timestamp(date)) or pd.Timestamp(date) > block.cutoff for date in dates):
            raise ValueError("block evidence was not available at cutoff")
        if (metadata.get("covariance_observation_date") is not None
                and metadata.get("covariance_available_date") is not None
                and pd.Timestamp(metadata["covariance_observation_date"])
                > pd.Timestamp(metadata["covariance_available_date"])):
            raise ValueError("covariance observation must precede availability")
    assets = first.loadings.index.append([block.loadings.index for block in blocks[1:]])
    if not assets.is_unique:
        raise ValueError("composed asset blocks must not overlap")
    return assets


def _concatenate(blocks: list[FactorModelSnapshot], name: str, order: pd.Index, *, axis: int = 0) -> object:
    """Join supplied evidence, leaving genuinely absent rows/cells unavailable."""
    parts = [getattr(block, name) for block in blocks]
    present = [part for part in parts if part is not None and len(part)]
    if not present:
        return None
    result = pd.concat(present, axis=axis, sort=axis == 1)
    return result.reindex(columns=order) if axis else result.reindex(order)


def compose_snapshots(
    snapshots: Sequence[FactorModelSnapshot], *, assets: Sequence[str] | None = None,
    loadings: pd.DataFrame | None = None, clusters: pd.Series | None = None,
    linkage_source: int | None = None, provenance: dict | None = None,
) -> FactorModelSnapshot:
    """Compose compatible orthogonal asset blocks, retaining cross-block factor risk.

    Args:
        snapshots: Nonempty disjoint blocks with the same cutoff, F, factor order,
            native coordinates, declared currency/units and orthogonal variance weight.
            Prepared correlated-residual evidence is unsupported without an explicit
            cross-block model; it is rejected rather than filled with zeros.
        assets: Exact ordered union of all input assets; omitted retains block order.
        loadings: Optional complete asset-by-factor override, in the output order.
            Cash zeroing or other economic choices must be supplied by the caller.
        clusters: Optional complete output memberships, including any caller-defined
            synthetic labels. Otherwise supplied memberships are concatenated.
        linkage_source: Explicit block whose original tree/cutoffs remain the main
            display evidence. Required when any block has a tree. Other blocks'
            original clustering evidence remains recorded in composition provenance.
        provenance: Additional plain application metadata; reserved block declarations
            and the composition record cannot be overwritten.

    Returns:
        A completed snapshot assembled by FL from the combined loadings and raw
        marginal variances. No R-squared, alpha or residual histories are invented.
        Distinct observation/availability declarations remain attached to each block.

    Raises:
        ValueError: If identities, policies, axes, dates or evidence are incompatible.
    """
    blocks = list(snapshots)
    union = _check_blocks(blocks)
    order = union if assets is None else selected_assets(assets, union)
    if len(order) != len(union):
        raise ValueError("composition assets must be the exact union; select a subset afterwards")
    has_tree = any(block.linkages is not None and len(block.linkages) for block in blocks)
    if has_tree and linkage_source is None:
        raise ValueError("linkage_source is required to choose the retained tree")
    if linkage_source is not None and (type(linkage_source) is not int or not 0 <= linkage_source < len(blocks)):
        raise ValueError("linkage_source must identify an input block")
    source = None if linkage_source is None else blocks[linkage_source]
    combined_loadings = _concatenate(blocks, "loadings", order) if loadings is None else loadings.copy()
    if (not combined_loadings.index.equals(order)
            or not combined_loadings.columns.equals(blocks[0].loadings.columns)):
        raise ValueError("loading override must match exact output asset/factor order")
    combined_clusters = _concatenate(blocks, "clusters", order) if clusters is None else clusters.copy()
    if combined_clusters is not None and not combined_clusters.index.equals(order):
        raise ValueError("cluster override must match exact output asset order")
    declarations = {key: blocks[0].provenance[key] for key in
                    ("reference_ccy", "asset_return_type", "factor_covariance_units")
                    if key in blocks[0].provenance}
    supplied = deepcopy(provenance or {})
    if "composition" in supplied or any(key in supplied and supplied[key] != value
                                        for key, value in declarations.items()):
        raise ValueError("provenance cannot replace composition evidence or common declarations")
    metadata = {**supplied, **declarations, "composition": dict(
        linkage_source=linkage_source, loadings_overridden=loadings is not None,
        clusters_overridden=clusters is not None,
        blocks=[dict(assets=block.loadings.index, provenance=block.provenance,
                     clusters=block.clusters, linkages=block.linkages, cutoffs=block.cutoffs,
                     clustering_source_assets=block.clustering_source_assets) for block in blocks],
    )}
    raw = CurrentFactorCovarData(
        x_covar=blocks[0].factor_covariance, y_betas=combined_loadings,
        y_variances=_concatenate(blocks, "fit_statistics", order), estimation_date=blocks[0].cutoff,
        residuals=_concatenate(blocks, "residuals", order, axis=1),
        residual_metadata=_concatenate(blocks, "residual_metadata", order),
        derived_signs=_concatenate(blocks, "derived_signs", order), clusters=combined_clusters,
        linkages=None if source is None else source.linkages,
        cutoffs=None if source is None else source.cutoffs,
    )
    completed = _materialize(
        raw, cutoff=blocks[0].cutoff, residual_type="orthogonal",
        residual_var_weight=blocks[0].residual_var_weight,
        factor_return_types=blocks[0].factor_return_types, factor_groups=blocks[0].factor_groups,
        provenance=metadata,
    )
    state = deepcopy(completed._fields)
    # The selected main tree still refers to its original fitted leaf universe,
    # even when synthetic output memberships have been supplied by the caller.
    state["clustering_source_assets"] = None if source is None else source.clustering_source_assets
    return FactorModelSnapshot(**state, cutoff=completed.cutoff,
                               residual_var_weight=completed.residual_var_weight)
