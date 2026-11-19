"""Independent contracts for native selection, composition, variants and storage."""
from copy import deepcopy
from dataclasses import replace
import pickle

import factorlasso as fl
import numpy as np
import pandas as pd
import pytest
from qis import ReturnTypes

import factormodels as fm


def raw_fit(date="2024-03-31", correlation=True):
    """Declare a nonuniform full-universe target that changes if recomputed on a subset."""
    assets = pd.Index(["Z", "A", "M"], name="asset")
    factors = pd.Index(["f", "g"], name="factor")
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    residuals = pd.DataFrame([[.1, .2, -.1], [.2, -.1, .3]], index=dates, columns=assets)
    metadata = pd.DataFrame(dict(frequency="ME", beta_span=6., annualisation_factor=12., residual_scale=12.),
                            index=assets)
    prepared = None if not correlation else fl.ResidualCorrelationData(
        correlation=pd.DataFrame([[1., .5, .1], [.5, 1., .3], [.1, .3, 1.]], index=assets, columns=assets),
        residual_returns=residuals / 12., asset_metadata=metadata,
        frequency="ME", span=6., observation_date=dates[-1], estimation_date=pd.Timestamp(date),
    )
    return fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[.04, .01], [.01, .09]], index=factors, columns=factors),
        y_betas=pd.DataFrame([[1., .2], [.3, .8], [-.2, .4]], index=assets, columns=factors),
        y_variances=pd.DataFrame(dict(residual_var=[.01, .04, .09], r2=[.8, .7, .2],
                                      insample_alpha=[.01, -.01, .03]), index=assets),
        estimation_date=pd.Timestamp(date), residuals=residuals, residual_metadata=metadata,
        residual_correlation=prepared, clusters=pd.Series("ME:1", index=assets, name="cluster"),
        linkages=pd.DataFrame([[0., 1., .2, 2.], [2., 3., .8, 3.]],
                             columns=["left", "right", "distance", "n_samples"],
                             index=["ME:step_0", "ME:step_1"]),
        cutoffs=pd.Series({"ME": .6}),
        derived_signs=pd.DataFrame([[1., np.nan], [1., 1.], [-1., 1.]], index=assets, columns=factors),
    )


def snapshot(raw=None, kind="empirical", weight=.35, rho=.6):
    """Materialize the synthetic original fit under an explicit selected policy."""
    raw = raw_fit() if raw is None else raw
    return fm.snapshots_from_factorlasso(
        raw, residual_type=kind, residual_var_weight=weight,
        residual_corr_weight=1. if kind == "orthogonal" else rho,
        provenance={"reference_ccy": "USD", "asset_return_type": "log",
                    "factor_covariance_units": "annual_native_coordinates"},
    ).snapshots[raw.estimation_date]


def test_selection_is_exact_without_assembly_and_owns_all_selected_fields(monkeypatch):
    """Selection is a principal view of fitted risk, including targets and inference units."""
    fitted = snapshot()
    selected = ["M", "Z"]

    def forbidden(*args, **kwargs):
        """Reject hidden numerical assembly during plain result operations."""
        raise AssertionError("selection and reload must never assemble or fit")

    monkeypatch.setattr(fl.CurrentFactorCovarData, "get_y_covar", forbidden)
    result = fitted.select_assets(selected)
    for name in ["asset_covariance", "residual_covariance", "raw_diagonal_reference_covariance",
                 "residual_correlation"]:
        pd.testing.assert_frame_equal(getattr(result, name), getattr(fitted, name).loc[selected, selected],
                                      check_exact=True)
    for name in ["loadings", "fit_statistics", "fit_diagnostics", "residual_metadata", "derived_signs"]:
        pd.testing.assert_frame_equal(getattr(result, name), getattr(fitted, name).loc[selected], check_exact=True)
    pd.testing.assert_series_equal(result.clusters, fitted.clusters.loc[selected])
    pd.testing.assert_frame_equal(result.residuals, fitted.residuals[selected])
    pd.testing.assert_frame_equal(result.linkages, fitted.linkages)
    pd.testing.assert_index_equal(result.clustering_source_assets["ME"], fitted.loadings.index)
    pd.testing.assert_index_equal(result.residual_target_source_assets, fitted.loadings.index)
    info = result.residual_correlation_metadata
    assert info["residual_returns"].columns.tolist() == selected
    assert info["asset_metadata"].index.tolist() == selected
    assert result.clusters_by_frequency["ME"].index.tolist() == selected
    second = result.select_assets(["Z"])
    restored = fm.model_from_payload(pickle.loads(pickle.dumps(fm.model_to_payload(second))))
    pd.testing.assert_frame_equal(restored.asset_covariance, fitted.asset_covariance.loc[["Z"], ["Z"]],
                                  check_exact=True)
    changed = result.residual_cluster_targets
    changed["exposure_cluster"].iloc[0, 1] = 99.
    assert result.residual_cluster_targets["exposure_cluster"].iloc[0, 1] != 99.
    payload = fm.model_to_payload(restored)
    payload["state"]["residual_metadata"].iloc[0, 0] = "changed"
    assert restored.residual_metadata.frequency.iloc[0] == "ME"


@pytest.mark.parametrize("assets,error", [([], ValueError), (["Z", "Z"], ValueError),
                                         (["missing"], KeyError), ("Z", TypeError), ({"Z": "new"}, TypeError)])
def test_selection_rejects_ambiguous_or_missing_assets(assets, error):
    """Reject missing identities and requests without a unique ordered universe."""
    with pytest.raises(error):
        snapshot().select_assets(assets)


def test_causal_date_access_preserves_original_cutoff_and_selected_universe():
    """Use only completed past snapshots and retain their original estimation date."""
    early, late = snapshot(raw_fit("2024-02-29")), snapshot(raw_fit("2024-04-30"))
    model = fm.RollingFactorModel({late.cutoff: late, early.cutoff: early}).select_assets(iter(["A", "Z"]))
    selected = model.at_or_before("2024-03-31")
    assert selected.cutoff == early.cutoff
    pd.testing.assert_frame_equal(selected.asset_covariance, early.asset_covariance.loc[["A", "Z"], ["A", "Z"]])
    assert model.at_or_before("2024-04-30").cutoff == late.cutoff
    assert model.at_or_before("2024-12-31").cutoff == late.cutoff
    for date in ["2024-01-31", pd.NaT]:
        with pytest.raises(ValueError):
            model.at_or_before(date)


@pytest.mark.parametrize("kind", ["orthogonal", "empirical", "exposure_cluster", "residual_cluster"])
@pytest.mark.parametrize("weight", [0., .35, 1.])
def test_variants_after_selection_and_payload_reload_match_full_fl_target(kind, weight):
    """A two-asset selection retains the three-asset cluster average, not its own pair."""
    raw = raw_fit()
    fitted = snapshot(raw)
    names = ["A", "Z"]
    selected = fm.model_from_payload(pickle.loads(pickle.dumps(fm.model_to_payload(fitted.select_assets(names)))))
    rho = 1. if kind == "orthogonal" else .6
    result = fm.prepare_residual_variant(selected, residual_type=kind,
                                         residual_var_weight=weight, residual_corr_weight=rho)
    expected = raw.get_y_covar(assets=names, residual_type=kind,
                               residual_var_weight=weight, residual_corr_weight=rho)
    pd.testing.assert_frame_equal(result.asset_covariance, expected, check_exact=True)
    pd.testing.assert_frame_equal(result.fit_statistics, fitted.fit_statistics.loc[names], check_exact=True)
    pd.testing.assert_frame_equal(result.fit_diagnostics, fitted.fit_diagnostics.loc[names], check_exact=True)
    pd.testing.assert_frame_equal(result.raw_diagonal_reference_covariance,
                                  fitted.raw_diagonal_reference_covariance.loc[names, names], check_exact=True)
    assert raw._residual_cluster_targets is None
    if kind == "exposure_cluster" and weight > 0:
        assert result.residual_cluster_targets[kind].iloc[0, 1] == pytest.approx(.3)
        assert result.residual_cluster_targets[kind].iloc[0, 1] != raw.residual_correlation.correlation.loc["A", "Z"]


def test_imported_filtered_fl_targets_survive_without_invented_original_identities():
    """Keep imported targets while refusing unsupported original leaf identities."""
    raw = raw_fit().filter_on_tickers(["A", "Z"])
    imported = snapshot(raw)
    assert imported.residual_target_source_assets is None
    with pytest.raises(ValueError, match="original leaf"):
        imported.select_assets(["A"])
    result = fm.prepare_residual_variant(imported, residual_type="exposure_cluster")
    pd.testing.assert_frame_equal(result.asset_covariance, raw.get_y_covar(residual_type="exposure_cluster"),
                                  check_exact=True)


def test_declared_burn_in_is_causal_and_later_missing_correlation_fails():
    """Permit explicit initial burn-in but never hide a later correlation gap."""
    early = snapshot(raw_fit("2024-02-29", correlation=False), kind="orthogonal")
    late = snapshot(raw_fit("2024-03-31"))
    missing = snapshot(raw_fit("2024-04-30", correlation=False), kind="orthogonal")
    model = fm.RollingFactorModel({early.cutoff: early, late.cutoff: late})
    with pytest.raises(ValueError, match="outside declared burn-in"):
        fm.prepare_residual_variant(model, residual_type="empirical")
    result = fm.prepare_residual_variant(model, residual_type="empirical", allow_orthogonal_burn_in=True)
    assert result.at_or_before(early.cutoff).residual_type == "orthogonal"
    assert result.at_or_before(late.cutoff).residual_type == "empirical"
    assert result.at_or_before(early.cutoff).provenance["residual_correlation_burn_in"]
    broken = fm.RollingFactorModel({**model.snapshots, missing.cutoff: missing})
    broken = fm.model_from_payload(fm.model_to_payload(broken.select_assets(["Z"])))
    with pytest.raises(ValueError, match="outside declared burn-in"):
        fm.prepare_residual_variant(broken, residual_type="empirical", allow_orthogonal_burn_in=True)


def test_missing_retained_target_is_not_reestimated_on_the_selected_universe():
    """Refuse to substitute subset cluster means for missing original targets."""
    fitted = snapshot().select_assets(["A", "Z"])
    payload = fm.model_to_payload(fitted)
    payload["state"]["residual_cluster_targets"] = None
    without = fm.model_from_payload(payload)
    with pytest.raises(ValueError, match="target was not retained"):
        fm.prepare_residual_variant(without, residual_type="exposure_cluster")


def composition_blocks():
    """Supply a main fitted block and a distinct cash/FX block with shared factors."""
    main = snapshot(raw_fit(correlation=False), kind="orthogonal", weight=1.).select_assets(["Z", "A"])
    assets = pd.Index(["CASH", "FX"], name="asset")
    cash = fm.snapshot_from_moments(
        pd.DataFrame([[1e-9, .3], [0., .5]], index=assets, columns=main.loadings.columns),
        main.factor_covariance, pd.Series([.001, .002], index=assets), date=main.cutoff,
        factor_return_types=main.factor_return_types, provenance=main.provenance,
    )
    return main, cash


def test_composition_includes_cross_block_factor_risk_and_explicit_cash_overrides():
    """Match independently declared moments and retain the caller's cash policy."""
    main, cash = composition_blocks()
    order = ["FX", "Z", "CASH", "A"]
    loadings = pd.concat([main.loadings, cash.loadings]).loc[order]
    loadings.loc["CASH"] = 0.
    labels = pd.Series(["cash", "ME:1", "cash", "ME:1"], index=loadings.index, name="cluster")
    result = fm.compose_snapshots([main, cash], assets=order, loadings=loadings,
                                  clusters=labels, linkage_source=0)
    # Independently assembled explicit moments; the FX/Z cross-block is nonzero.
    b = np.array([[0., .5], [1., .2], [0., 0.], [.3, .8]])
    expected = b @ np.array([[.04, .01], [.01, .09]]) @ b.T + np.diag([.002, .01, .001, .04])
    np.testing.assert_allclose(result.asset_covariance, expected, rtol=0, atol=1e-16)
    assert result.asset_covariance.loc["FX", "Z"] == pytest.approx(.014)
    assert result.loadings.loc["CASH"].eq(0).all()
    assert pd.isna(result.fit_statistics.loc["CASH", "r2"])
    assert pd.isna(result.fit_statistics.loc["FX", "insample_alpha"])
    assert result.residuals[["CASH", "FX"]].isna().all().all()
    pd.testing.assert_frame_equal(result.linkages, main.linkages)
    pd.testing.assert_index_equal(result.clustering_source_assets["ME"], pd.Index(["Z", "A", "M"], name="asset"))
    assert result.provenance["composition"]["loadings_overridden"]
    restored = fm.model_from_payload(fm.model_to_payload(result))
    pd.testing.assert_frame_equal(restored.asset_covariance, result.asset_covariance, check_exact=True)


@pytest.mark.parametrize("fault", ["overlap", "cutoff", "factors", "covariance", "policy", "currency", "future",
                                  "missing_tree_choice", "partial_union", "loading_order"])
def test_composition_rejects_incompatible_or_ambiguous_blocks(fault):
    """Fail on incompatible evidence before assembling cross-block covariance."""
    main, cash = composition_blocks()
    options = dict(linkage_source=0)
    state = fm.model_to_payload(cash)
    if fault == "overlap":
        cash = main
    elif fault == "cutoff":
        state["state"]["cutoff"] += pd.Timedelta(days=1)
    elif fault == "factors":
        state["state"]["factor_return_types"]["f"] = ReturnTypes.DIFFERENCE
    elif fault == "covariance":
        state["state"]["factor_covariance"] *= 2
    elif fault == "policy":
        state["state"]["residual_type"] = "empirical"
    elif fault == "currency":
        state["state"]["provenance"]["reference_ccy"] = "EUR"
    elif fault == "future":
        state["state"]["provenance"]["covariance_available_date"] = pd.Timestamp("2025-01-01")
    elif fault == "missing_tree_choice":
        options = {}
    elif fault == "partial_union":
        options["assets"] = ["Z", "A"]
    elif fault == "loading_order":
        options["loadings"] = pd.concat([cash.loadings, main.loadings])
    with pytest.raises(ValueError):
        if fault not in ("overlap",):
            cash = fm.model_from_payload(state)
        fm.compose_snapshots([main, cash], **options)


@pytest.mark.parametrize("fault", ["version", "missing", "extra", "duplicate_date", "future_correlation", "fl_object"])
def test_payload_rejects_incomplete_or_unsupported_evidence(fault):
    """Require the complete versioned native contract without FL instances."""
    fitted = snapshot()
    payload = fm.model_to_payload(fitted)
    if fault == "version":
        payload["schema_version"] = 2
    elif fault == "missing":
        del payload["state"]["residual_cluster_targets"]
    elif fault == "extra":
        payload["state"]["invented"] = 1
    elif fault == "duplicate_date":
        payload = dict(schema_version=1, model_type="rolling", snapshots=[payload, deepcopy(payload)])
    elif fault == "future_correlation":
        payload["state"]["residual_correlation_metadata"]["estimation_date"] = pd.Timestamp("2025-01-01")
    elif fault == "fl_object":
        payload["state"]["provenance"]["invalid"] = raw_fit()
    with pytest.raises((ValueError, TypeError)):
        fm.model_from_payload(payload)


def test_filtered_target_counterexample_changes_risk_if_original_target_is_lost():
    """Independently demonstrate the defect that the retained-target contract prevents."""
    raw = raw_fit()
    selected = raw.filter_on_tickers(["A", "Z"])
    defective = replace(selected)
    assert not selected.get_y_covar(residual_type="exposure_cluster").equals(
        defective.get_y_covar(residual_type="exposure_cluster"))


def test_warmup_missing_membership_keeps_declared_tree_leaves():
    """Cadence metadata identifies a zeroed leaf without creating a cluster label."""
    raw = raw_fit()
    raw.clusters.iloc[1] = np.nan
    fitted = snapshot(raw)
    selected = fitted.select_assets(['M', 'Z'])
    pd.testing.assert_index_equal(selected.clustering_source_assets['ME'], raw.clusters.index)
    pd.testing.assert_frame_equal(selected.linkages, raw.linkages)
    assert pd.isna(fitted.clusters.iloc[1])


def test_missing_membership_without_cadence_does_not_guess_tree_leaves():
    """Historical inputs without original identity evidence remain unselectable."""
    raw = raw_fit()
    raw.clusters.iloc[1] = np.nan
    raw = replace(raw, residual_metadata=None, residual_correlation=None)
    fitted = snapshot(raw, kind="orthogonal")
    with pytest.raises(ValueError, match='original leaf assets'):
        fitted.select_assets(['M', 'Z'])


def test_native_lineage_labels_match_fl_and_are_independent():
    """The statistical owner supplies exactly the same labels over a changing history."""
    first, last = raw_fit('2024-03-31'), raw_fit('2024-04-30')
    last.clusters.iloc[-1] = 'ME:2'
    raw = {first.estimation_date: first, last.estimation_date: last}
    native = fm.snapshots_from_factorlasso(raw, residual_type='orthogonal')
    expected = fl.analyze_cluster_lineage(fl.RollingFactorCovarData(data=raw)).to_label_panel()
    actual = fm.prepare_cluster_labels(native)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    actual.iloc[0, 0] = 'changed'
    pd.testing.assert_frame_equal(fm.prepare_cluster_labels(native), expected, check_exact=True)
