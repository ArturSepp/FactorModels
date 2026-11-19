"""Prepare plain evidence needed by native selection and residual variants.

Purpose: Retain original-universe residual targets and dendrogram leaf identities.
Inputs: A dated FL intermediate, before the FM result is exposed to consumers.
Outputs: Plain target matrices and original ordered asset identities.
Units: Targets are dimensionless correlations; leaf identities are categorical.
Conditioning: FL computes targets on its full fitted universe before selection.
Ownership: FL owns clustering; FM retains exact outputs without storing an FL object.
"""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
from factorlasso import CurrentFactorCovarData

__all__ = []


def residual_target_evidence(data: CurrentFactorCovarData) -> tuple[dict | None, pd.Index | None]:
    """Use FL's public selection operation to materialize its canonical block targets.

    FL exposes no separate public target getter. Its selected result's private
    plain cache is copied at this estimation boundary only; no algorithm is copied.
    An already-filtered FL input has retained target values but unknown original
    identities, which must not be fabricated from its smaller current axis.
    """
    if data.residual_correlation is None:
        return None, None
    original = data._residual_cluster_targets
    selected = data.filter_on_tickers(data.y_betas.index.tolist())
    return deepcopy(selected._residual_cluster_targets), (data.y_betas.index.copy() if original is None else None)


def clustering_leaf_evidence(
    clusters: dict | None, linkages: dict | None, *,
    memberships: pd.Series | None = None, residual_metadata: pd.DataFrame | None = None,
) -> dict[str, pd.Index] | None:
    """Record only leaf orders supported by the complete membership/tree dimensions."""
    if clusters is None:
        return None
    result = {}
    for frequency in dict.fromkeys([*clusters, *(linkages or {})]):
        labels = clusters.get(frequency, pd.Series(dtype=object))
        identities = labels.index
        # Warm-up zeroing clears membership, not the fitted tree's leaf. The
        # producer's own cadence metadata identifies those leaves in their
        # original flattened membership order; never infer a missing cadence.
        if memberships is not None and residual_metadata is not None and 'frequency' in residual_metadata:
            mask = memberships.astype(str).str.startswith(f'{frequency}:')
            mask |= memberships.isna() & residual_metadata['frequency'].reindex(memberships.index).eq(frequency)
            identities = memberships.index[mask]
        tree = (linkages or {}).get(frequency)
        if tree is None or len(tree) == max(0, len(identities) - 1):
            result[frequency] = identities.copy()
    return result or None
