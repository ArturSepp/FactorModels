"""Deterministic in-memory teaching inputs available from an installed package."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factormodels.bundle import BundleSettings, FactorBundle

__all__ = ["synthetic_bundle"]


def synthetic_bundle() -> FactorBundle:
    """Return fresh synthetic USD total-return NAVs and complete fit settings.

    The fixed seed produces 121 monthly observations from January 2014 through
    January 2024 for two factors and four assets. Settings declare decimal log
    returns and annual covariance. The inputs are illustrative, not market data.
    No acquisition, fitting, rendering or persistence occurs.

    Returns:
        FactorBundle: Independent histories, metadata and configuration ready for
        an explicit ``fit_bundle`` call.
    """
    rng = np.random.default_rng(6102026)
    dates = pd.date_range("2014-01-31", periods=121, freq="ME")
    x = rng.normal([.004, .001], [.035, .018], (120, 2))
    b = np.array([[.9, .1], [.1, .9], [.6, .4], [.4, -.2]])
    y = x @ b.T + rng.normal(0., .013, (120, 4))
    levels = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(2), x.cumsum(axis=0)])),
                          index=dates, columns=["Equity", "Rates"])
    prices = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(4), y.cumsum(axis=0)])),
                          index=dates, columns=["Asset A", "Asset B", "Asset C", "Asset D"])
    metadata = pd.DataFrame({"currency": "USD", "hedge_ratio": 0., "return_frequency": "ME"},
                            index=prices.columns)
    return FactorBundle(
        factor_levels=levels, fx_spots=pd.DataFrame({"USD": 1.}, index=dates),
        domestic_rates=pd.DataFrame({"USD": 0.}, index=dates), asset_prices=prices,
        asset_metadata=metadata,
        settings=BundleSettings(reference_ccy="USD", factor_names=tuple(levels.columns),
                                estimation_start=dates[36], estimation_end=dates[-1],
                                recipe_id="synthetic-reference-v1"),
        provenance={"data_kind": "synthetic", "seed": 6102026})
