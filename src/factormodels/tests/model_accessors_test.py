"""Completed-table accessor contracts for installed fitted models."""
from copy import deepcopy

import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData
from qis.perfstats.config import ReturnTypes

import factormodels as tf


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
