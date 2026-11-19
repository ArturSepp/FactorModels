"""Ordered selection of completed model fields without numerical assembly.

Purpose: Slice owned result tables while retaining original clustering evidence.
Inputs: Completed field mappings and a nonempty ordered asset selection.
Outputs: Constructor fields with exact principal covariance submatrices.
Units: Annual moments and native residual coordinates are unchanged.
Conditioning: Selection never refits, re-cuts a tree or rebuilds residual targets.
Ownership: FM owns result selection; this module has no estimator dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import pandas as pd

__all__ = []


def selected_assets(assets: Sequence[str], available: pd.Index) -> pd.Index:
    """Validate a strict ordered selection, without implicit renaming or filling."""
    if isinstance(assets, (str, bytes, Mapping)):
        raise TypeError("assets must be an ordered sequence, not a string or rename mapping")
    names = list(assets)
    index = pd.Index(names, name=available.name)
    if index.empty or not index.is_unique:
        raise ValueError("asset selection must be nonempty and unique")
    if not index.isin(available).all():
        raise KeyError(f"unknown assets: {index[~index.isin(available)].tolist()}")
    return available.take(available.get_indexer(index))


def select_snapshot_fields(state: dict, assets: Sequence[str]) -> dict:
    """Copy all asset-indexed fields and preserve original-universe evidence."""
    index = selected_assets(assets, state["loadings"].index)
    names = index.tolist()
    source = state["clustering_source_assets"] or {}
    trees = state["linkages_by_frequency"] or {}
    if (set(trees) - set(source) or
            (state["linkages"] is not None and len(state["linkages"]) and not source)):
        raise ValueError("selection requires original leaf assets for retained linkage evidence")
    result = deepcopy(state)
    for name in ("loadings", "fit_statistics", "fit_diagnostics", "residual_metadata",
                 "derived_signs", "clusters"):
        if result[name] is not None:
            result[name] = result[name].loc[names].copy()
    for name in ("asset_covariance", "residual_covariance", "raw_diagonal_reference_covariance",
                 "residual_correlation"):
        if result[name] is not None:
            result[name] = result[name].loc[names, names].copy()
    if result["residuals"] is not None:
        result["residuals"] = result["residuals"].loc[:, names].copy()
    metadata = result["residual_correlation_metadata"]
    if metadata is not None:
        metadata["residual_returns"] = metadata["residual_returns"].loc[:, names].copy()
        metadata["asset_metadata"] = metadata["asset_metadata"].loc[names].copy()
    if result["clusters_by_frequency"] is not None:
        result["clusters_by_frequency"] = {
            freq: labels.loc[index[index.isin(labels.index)].tolist()].copy()
            for freq, labels in result["clusters_by_frequency"].items()
        }
    if result["residual_cluster_targets"] is not None:
        result["residual_cluster_targets"] = {
            kind: matrix.loc[names, names].copy()
            for kind, matrix in result["residual_cluster_targets"].items()
        }
    return result


def rename_snapshot_fields(state: dict, mapping: Mapping[str, str]) -> dict:
    """Rename explicit asset identities on every typed axis, preserving tree order."""
    if not isinstance(mapping, Mapping):
        raise TypeError("asset rename requires a mapping")
    available = state["loadings"].index
    if set(mapping) - set(available) or any(not isinstance(v, str) or not v for v in mapping.values()):
        raise ValueError("asset rename requires known keys and nonempty string labels")
    universe = set(available)
    for leaves in (state["clustering_source_assets"] or {}).values():
        universe.update(leaves)
    if state["residual_target_source_assets"] is not None:
        universe.update(state["residual_target_source_assets"])
    if len({mapping.get(v, v) for v in universe}) != len(universe):
        raise ValueError("asset rename collides with a retained fitted identity")
    result = deepcopy(state)
    for name in ("loadings", "fit_statistics", "fit_diagnostics", "residual_metadata", "derived_signs", "clusters"):
        if result[name] is not None:
            result[name] = result[name].rename(index=mapping)
    for name in ("asset_covariance", "residual_covariance", "raw_diagonal_reference_covariance",
                 "residual_correlation"):
        if result[name] is not None:
            result[name] = result[name].rename(index=mapping, columns=mapping)
    if result["residuals"] is not None:
        result["residuals"] = result["residuals"].rename(columns=mapping)
    metadata = result["residual_correlation_metadata"]
    if metadata is not None:
        metadata["residual_returns"] = metadata["residual_returns"].rename(columns=mapping)
        metadata["asset_metadata"] = metadata["asset_metadata"].rename(index=mapping)
    if result["clusters_by_frequency"] is not None:
        result["clusters_by_frequency"] = {
            freq: labels.rename(index=mapping) for freq, labels in result["clusters_by_frequency"].items()}
    if result["residual_cluster_targets"] is not None:
        result["residual_cluster_targets"] = {
            kind: matrix.rename(index=mapping, columns=mapping)
            for kind, matrix in result["residual_cluster_targets"].items()}
    if result["clustering_source_assets"] is not None:
        result["clustering_source_assets"] = {
            freq: labels.map(lambda label: mapping.get(label, label))
            for freq, labels in result["clustering_source_assets"].items()}
    if result["residual_target_source_assets"] is not None:
        result["residual_target_source_assets"] = result["residual_target_source_assets"].map(
            lambda label: mapping.get(label, label))
    # Observation dates are keyed by return cadence, not asset identity.
    result["provenance"].setdefault("asset_label_renames", []).append(dict(mapping))
    return result
