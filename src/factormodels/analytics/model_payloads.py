"""Versioned plain-field payloads for completed factor models.

Purpose: Share one strict native-result reconstruction contract between applications.
Inputs: Completed FM results or schema-versioned mappings of owned pandas/NumPy fields.
Outputs: Independent payload mappings or validated native model objects.
Units: Annual covariance and producer-declared residual coordinates round-trip unchanged.
Conditioning: No fitting, covariance assembly, default inference or historical FL reconstruction.
Ownership: FM owns the result schema and optional portable archive; applications own paths and workbook policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel

__all__ = ["model_to_payload", "model_from_payload"]

_FIELDS = frozenset({
    "loadings", "factor_covariance", "residual_covariance", "asset_covariance",
    "raw_diagonal_reference_covariance", "fit_statistics", "fit_diagnostics",
    "residuals", "residual_metadata", "residual_correlation", "residual_correlation_metadata",
    "clusters", "linkages", "cutoffs", "derived_signs", "clusters_by_frequency",
    "linkages_by_frequency", "cutoffs_by_frequency", "clustering_source_assets",
    "residual_cluster_targets", "residual_target_source_assets", "factor_return_types",
    "factor_groups", "provenance", "cutoff", "residual_type", "residual_var_weight",
    "residual_corr_weight",
})


def model_to_payload(model: FactorModelSnapshot | RollingFactorModel) -> dict:
    """Copy a completed model into schema 1, retaining every table and identity.

    The mapping contains native pandas/NumPy objects and scalar metadata; it is
    not a JSON string or an Excel writer. A caller's storage layer must preserve
    index types/names, order, dtypes, missing masks and nested values losslessly.
    Loading performs no dynamic class imports and retains no FactorLasso object.

    Args:
        model: Completed snapshot or rolling model to export without refitting.

    Returns:
        Independent schema-1 mapping with complete typed tables and metadata.
    """
    if isinstance(model, RollingFactorModel):
        return dict(schema_version=1, model_type="rolling",
                    snapshots=[model_to_payload(snapshot) for snapshot in model.snapshots.values()])
    if not isinstance(model, FactorModelSnapshot):
        raise TypeError("model must be FactorModelSnapshot or RollingFactorModel")
    state = deepcopy(model._fields)
    state.update(cutoff=model.cutoff, residual_type=model.residual_type,
                 residual_var_weight=model.residual_var_weight, residual_corr_weight=model.residual_corr_weight)
    if set(state) != _FIELDS:
        raise ValueError("result fields differ from the versioned schema")
    return dict(schema_version=1, model_type="snapshot", state=state)


def model_from_payload(payload: Mapping) -> FactorModelSnapshot | RollingFactorModel:
    """Validate a complete schema-1 payload without filling missing evidence.

    Missing/extra fields, unsupported versions, duplicate dates and invalid native
    results fail explicitly. This consumes trusted, already-decoded tables; file
    decoding and security belong to the application's selected storage format.

    Args:
        payload: Trusted decoded schema-1 mapping with every required field.

    Returns:
        Validated native snapshot or rolling model with independent owned fields.
    """
    if not isinstance(payload, Mapping) or type(payload.get("schema_version")) is not int:
        raise ValueError("model payload requires an integer schema_version")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported model payload schema_version")
    if payload.get("model_type") == "snapshot":
        if set(payload) != {"schema_version", "model_type", "state"}:
            raise ValueError("invalid snapshot payload fields")
        state = payload["state"]
        if not isinstance(state, Mapping) or set(state) != _FIELDS:
            raise ValueError("snapshot payload must contain every schema field exactly")
        return FactorModelSnapshot(**deepcopy(dict(state)))
    if payload.get("model_type") == "rolling":
        if set(payload) != {"schema_version", "model_type", "snapshots"}:
            raise ValueError("invalid rolling payload fields")
        if not isinstance(payload["snapshots"], list) or not payload["snapshots"]:
            raise ValueError("rolling payload requires a nonempty snapshot list")
        snapshots = [model_from_payload(item) for item in payload["snapshots"]]
        if not all(isinstance(item, FactorModelSnapshot) for item in snapshots):
            raise ValueError("rolling payload entries must be snapshots")
        dates = [item.cutoff for item in snapshots]
        if len(set(dates)) != len(dates):
            raise ValueError("rolling payload has duplicate snapshot dates")
        return RollingFactorModel(dict(zip(dates, snapshots)))
    raise ValueError("unknown model_type")
