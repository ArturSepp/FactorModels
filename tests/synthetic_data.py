"""Small deterministic, provider-independent factor-model fixtures."""

import numpy as np
import pandas as pd


def factor_panels(periods: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return factor prices and asset log returns from a fixed synthetic design."""
    rng = np.random.default_rng(20261018)
    index = pd.bdate_range("2023-01-02", periods=periods)
    factors = pd.DataFrame(rng.normal(0.0002, 0.007, (periods, 2)), index=index,
                           columns=["Equity", "Rates"])
    loadings = np.array([[0.8, 0.2], [0.2, 0.9], [0.5, 0.4]])
    assets = pd.DataFrame(factors.to_numpy() @ loadings.T + rng.normal(0.0, 0.002, (periods, 3)),
                          index=index, columns=["Asset A", "Asset B", "Asset C"])
    return 100.0 * np.exp(factors.cumsum()), assets
