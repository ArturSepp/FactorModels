"""Offline native result selection, composition and schema round-trip."""
from enum import Enum

import numpy as np
import pandas as pd

from factormodels import (
    RollingFactorModel, compose_snapshots, model_from_payload, model_to_payload,
    prepare_residual_variant, snapshot_from_moments,
)


class LocalTests(Enum):
    """Run the supplied-moment native result example."""

    NATIVE_RESULTS = 1


def run_local_test(case: LocalTests | int = LocalTests.NATIVE_RESULTS) -> None:
    """Verify exact selection/reload and independent cross-block covariance."""
    LocalTests(case)
    date = pd.Timestamp("2024-03-31")
    factors = pd.Index(["equity", "rates"], name="factor")
    covariance = pd.DataFrame([[.04, .01], [.01, .09]], index=factors, columns=factors)
    first = snapshot_from_moments(
        pd.DataFrame([[1., .2]], index=["asset"], columns=factors), covariance,
        pd.Series([.01], index=["asset"]), date=date,
    )
    second = snapshot_from_moments(
        pd.DataFrame([[0., .5]], index=["fx_cash"], columns=factors), covariance,
        pd.Series([.002], index=["fx_cash"]), date=date,
    )
    combined = compose_snapshots([first, second], assets=["fx_cash", "asset"])
    # The off-diagonal comes from common factors, despite disjoint fitted blocks.
    np.testing.assert_allclose(combined.asset_covariance, [[.0245, .014], [.014, .0576]],
                               rtol=0, atol=1e-16)
    selected = combined.select_assets(["asset"])
    pd.testing.assert_frame_equal(selected.asset_covariance,
                                  combined.asset_covariance.loc[["asset"], ["asset"]], check_exact=True)
    rolling = RollingFactorModel({date: combined})
    assert rolling.at_or_before("2024-04-15").cutoff == date
    restored = model_from_payload(model_to_payload(rolling))
    pd.testing.assert_frame_equal(restored.covariances[date], combined.asset_covariance, check_exact=True)
    reference = prepare_residual_variant(combined, residual_type="orthogonal")
    pd.testing.assert_frame_equal(reference.asset_covariance, combined.asset_covariance, check_exact=True)
    print("Native selection, cross-block covariance and schema round-trip verified.")


if __name__ == "__main__":
    run_local_test(case=LocalTests.NATIVE_RESULTS)
