"""Installed teaching inputs and completed-table accessor contracts."""
import builtins
from copy import deepcopy
import io
import socket

import numpy as np
import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData
from qis.perfstats.config import ReturnTypes

import factormodels as tf
import factormodels.bundle as bundle_module
from factormodels.examples import synthetic_bundle


def _snapshot() -> tf.FactorModelSnapshot:
    """Build known, nonalphabetically ordered mixed-coordinate fitted tables."""
    assets = pd.Index(["Z", "A", "M"], name="asset")
    factors = pd.Index(["Yield", "Equity"], name="factor")
    cutoff = pd.Timestamp("2024-01-31")
    return tf.FactorModelSnapshot(
        data=CurrentFactorCovarData(
            x_covar=pd.DataFrame([[.04, -.005], [-.005, .09]], index=factors, columns=factors),
            y_betas=pd.DataFrame([[-.1, .8], [-.4, .2], [.3, .5]], index=assets, columns=factors),
            y_variances=pd.DataFrame({"residual_var": [.01, .02, .03]}, index=assets),
            estimation_date=cutoff,
        ),
        cutoff=cutoff,
        factor_return_types={"Yield": ReturnTypes.DIFFERENCE, "Equity": ReturnTypes.LOG},
        provenance={"reference_ccy": "USD", "factor_covariance_units": "annual_native_coordinates"},
    )


@pytest.mark.parametrize("accessor,field", [("loadings", "y_betas"), ("factor_covariance", "x_covar")])
def test_snapshot_table_accessors_preserve_fit_and_return_independent_copies(accessor, field, monkeypatch):
    """Reading and editing displayed tables cannot refit or mutate the completed model."""
    snapshot = _snapshot()
    expected = getattr(snapshot.data, field).copy()
    covariance = snapshot.asset_covariance.copy()
    provenance = deepcopy(snapshot.provenance)
    coordinates = dict(snapshot.factor_return_types)

    def forbidden(*args, **kwargs):
        """Fail if inspecting completed data tries to invoke an estimator."""
        raise AssertionError("table access must not refit")

    monkeypatch.setattr(tf.FactorCovarEstimator, "fit_current", forbidden)
    monkeypatch.setattr(tf.FactorCovarEstimator, "fit_rolling", forbidden)
    monkeypatch.setattr(tf, "fit_bundle", forbidden)
    monkeypatch.setattr(bundle_module, "fit_bundle", forbidden)
    displayed = getattr(snapshot, accessor)
    second_read = getattr(snapshot, accessor)
    pd.testing.assert_frame_equal(displayed, expected, check_exact=True)
    displayed.iloc[0, 0] = 999.
    displayed.index.name = "changed assets"
    displayed.columns.name = "changed factors"
    pd.testing.assert_frame_equal(getattr(snapshot.data, field), expected, check_exact=True)
    pd.testing.assert_frame_equal(second_read, expected, check_exact=True)
    pd.testing.assert_frame_equal(getattr(snapshot, accessor), expected, check_exact=True)
    pd.testing.assert_frame_equal(snapshot.asset_covariance, covariance, check_exact=True)
    assert snapshot.cutoff == pd.Timestamp("2024-01-31")
    assert snapshot.provenance == provenance
    assert snapshot.factor_return_types == coordinates


def test_synthetic_factory_is_public_reproducible_and_returns_fresh_inputs():
    """Each call owns all histories, metadata and provenance with the same declared inputs."""
    assert tf.synthetic_bundle is synthetic_bundle
    assert "synthetic_bundle" in tf.__all__
    first, second = synthetic_bundle(), synthetic_bundle()
    frames = ("factor_levels", "asset_prices", "fx_spots", "domestic_rates", "asset_metadata")
    for name in frames:
        original = getattr(first, name)
        independent = getattr(second, name)
        pd.testing.assert_frame_equal(original, independent, check_exact=True)
        before = independent.copy()
        original.iloc[0, 0] = "changed" if name == "asset_metadata" else -999.
        original.index.name = "changed index"
        pd.testing.assert_frame_equal(independent, before, check_exact=True)
        pd.testing.assert_frame_equal(getattr(synthetic_bundle(), name), before, check_exact=True)
    first.provenance["seed"] = 0
    assert second.provenance == {"data_kind": "synthetic", "seed": 6102026}
    assert synthetic_bundle().provenance == second.provenance
    assert first.settings is not second.settings
    pd.testing.assert_frame_equal(first.settings.to_frame(), second.settings.to_frame(), check_exact=True)
    assert second.factor_levels.shape == (121, 2)
    assert second.asset_prices.shape == (121, 4)
    assert second.factor_levels.index[[0, -1]].tolist() == [pd.Timestamp("2014-01-31"), pd.Timestamp("2024-01-31")]
    assert second.settings.estimation_start == pd.Timestamp("2017-01-31")
    assert second.settings.estimation_end == pd.Timestamp("2024-01-31")
    assert second.settings.recipe_id == "synthetic-reference-v1"
    assert second.settings.reference_ccy == "USD"
    assert second.settings.is_log_returns
    assert second.settings.covariance_units == "annual_native_coordinates"
    assert second.settings.factor_names == ("Equity", "Rates")


def test_synthetic_factory_needs_no_fit_network_or_files_and_preserves_global_rng(monkeypatch):
    """Input creation is an isolated in-memory action, with a local random generator."""
    before = np.random.get_state()

    def forbidden(*args, **kwargs):
        """Reject hidden computation, acquisition or filesystem activity."""
        raise AssertionError("synthetic inputs must be produced only in memory")

    with monkeypatch.context() as patches:
        patches.setattr(tf.FactorCovarEstimator, "fit_current", forbidden)
        patches.setattr(tf.FactorCovarEstimator, "fit_rolling", forbidden)
        patches.setattr(tf, "fit_bundle", forbidden)
        patches.setattr(bundle_module, "fit_bundle", forbidden)
        patches.setattr(tf, "save_bundle", forbidden)
        patches.setattr(bundle_module, "save_bundle", forbidden)
        patches.setattr(socket.socket, "connect", forbidden)
        patches.setattr(socket, "create_connection", forbidden)
        patches.setattr(builtins, "open", forbidden)
        patches.setattr(io, "open", forbidden)
        patches.setattr(pd.DataFrame, "to_csv", forbidden)
        bundle = synthetic_bundle()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert bundle.factor_levels.shape == (121, 2)
