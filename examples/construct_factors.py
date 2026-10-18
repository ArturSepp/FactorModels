"""Construct completed factor inputs from deterministic synthetic daily prices.

No provider, Rosaa installation, files or network access is required. Construction
uses weekly log returns, annualised decimal volatility and quarterly decisions;
QIS holds units between decisions. These are teaching recipes, not market evidence.
"""

import numpy as np
import pandas as pd

from tradablefactors import (
    Factor,
    FactorFamily,
    RiskFactors,
    VolTargetPolicy,
    assemble_factor_model,
    build_factor_panel,
    compute_beta_loading_signs,
)


class TeachingFactors(RiskFactors):
    """Ordered identifiers for two independently usable synthetic factor NAVs."""

    GROWTH = "Growth basket"
    SPREAD = "Spread basket"


def run_local_test(case: int = 1):
    """Return a complete definition, NAV panel and aligned loading constraints."""
    if case != 1:
        raise ValueError("The construction example supports only case 1.")
    dates = pd.bdate_range("2023-01-02", periods=520)
    step = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame({
        "Source A": 100 * np.exp(.0002 * step + .04 * np.sin(step / 13)),
        "Source B": 100 * np.exp(.0001 * step + .03 * np.cos(step / 19)),
    }, index=dates)
    growth = FactorFamily("Growth", vol_target=.10, long_only_sign=1.)
    spread = FactorFamily("Spread", vol_target=.05, is_gated=True)
    catalogue = {
        TeachingFactors.GROWTH.value: Factor(
            name=TeachingFactors.GROWTH.value, family=growth,
            long_side_tickers=("Source A", "Source B"),
            long_side_weights=(.6, .4)),
        TeachingFactors.SPREAD.value: Factor(
            name=TeachingFactors.SPREAD.value, family=spread,
            long_side_tickers=("Source A",), long_side_weights=(1.,),
            short_side_tickers=("Source B",), short_side_weights=(1.,),
            vol_target_policy=VolTargetPolicy.NONE),
    }
    spec = assemble_factor_model(
        name="Synthetic teaching model", factors=TeachingFactors,
        file_name="synthetic_factors", catalogue=catalogue)
    panel = build_factor_panel(
        spec, prices, rebalancing_freq="QE", vol_span=None, returns_freq="W-WED")
    signs = compute_beta_loading_signs(
        spec, long_only_betas=pd.Series({"Asset A": True, "Asset B": False}),
        gated_factor_exposure=pd.Series({"Asset A": False, "Asset B": True}))
    assert list(panel.columns) == spec.factor_names()
    assert panel.notna().sum().min() > 300
    np.testing.assert_array_equal(signs.to_numpy(), [[1., 0.], [1., 1.]])
    print(panel.tail())
    print(signs)
    return spec, panel, signs


if __name__ == "__main__":
    run_local_test(case=1)
