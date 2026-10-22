"""Define factors, estimate rolling models and retrieve a dated covariance.

Run: python -m examples.minimal_model

Inputs are synthetic monthly USD log returns and total-return factor NAVs; no
cash/FX adjustment, network access or files are needed. Quarterly fits use only
history through each cutoff, with 36-observation EWMA spans (not fixed windows).
Covariances are annualised asset log-return covariances, including factor risk
and the explicitly selected orthogonal (diagonal) residual covariance.
"""
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType

from factormodels import (
    Factor, FactorCovarEstimator, FactorFamily, FactorModelSpec,
    RiskFactors, VolTargetPolicy, build_factor_panel,
)


class MinimalFactors(RiskFactors):
    """Ordered factor identifiers for this two-factor model."""

    EQUITY = "Equity"
    RATES = "Rates"


def run_local_test(case: int = 1):
    """Return the definition, rolling fit and selected-date asset covariance."""
    if case != 1:
        raise ValueError("The minimal model example supports only case 1.")

    # 1. Define each factor as a single-source basket, without volatility targeting.
    spec = FactorModelSpec(
        name="Minimal two-factor model", factors=MinimalFactors, file_name="minimal_model",
        baskets={
            member: Factor(
                name=member.value, family=FactorFamily(member.value, vol_target=None),
                long_side_tickers=(f"{member.value} source",), long_side_weights=(1.,),
                vol_target_policy=VolTargetPolicy.NONE,
            )
            for member in MinimalFactors
        },
    )

    # Synthetic monthly inputs; replace these two panels with your own histories.
    dates = pd.date_range("2015-12-31", "2024-12-31", freq="ME")
    rng = np.random.default_rng(42)
    factor_log_returns = rng.normal([.005, .002], [.04, .02], (len(dates) - 1, 2))
    source_prices = pd.DataFrame(
        100. * np.exp(np.vstack([np.zeros(2), factor_log_returns.cumsum(axis=0)])),
        index=dates, columns=["Equity source", "Rates source"],
    )
    # These generating loadings are only for simulation; the estimator fits its own.
    loadings = np.array([[.9, .1], [.1, .8], [.5, .4]])
    asset_returns = pd.DataFrame(
        factor_log_returns @ loadings.T + rng.normal(0., .01, (len(dates) - 1, 3)),
        index=dates[1:], columns=["Equity asset", "Bond asset", "Balanced asset"],
    )
    factor_prices = build_factor_panel(
        spec, source_prices, returns_freq="ME", rebalancing_freq="ME",
    )

    # 2. Estimate loadings and covariance at each quarter-end, using monthly inputs.
    estimator = FactorCovarEstimator(
        lasso_model=LassoModel(
            model_type=LassoModelType.LASSO, span=36, warmup_period=6,
            reg_lambda=1e-5, solver="CLARABEL",
        ),
        factor_returns_freq="ME", factor_covar_span=36,
        rebalancing_freq="QE", residual_type="orthogonal",
    )
    rolling = estimator.fit_rolling(
        risk_factor_prices=factor_prices, asset_returns_dict={"ME": asset_returns},
        time_period=qis.TimePeriod("2024-03-31", "2024-12-31"),
        assets=list(asset_returns.columns),
    )

    # 3. Retrieve the full asset covariance by an exact fitted date, or the latest date.
    covariances = rolling.covariances  # dict[pd.Timestamp, pd.DataFrame]
    as_of = pd.Timestamp("2024-06-30")
    covariance = covariances[as_of]
    latest_date = max(covariances)
    latest_covariance = covariances[latest_date]
    assert covariance.index.equals(asset_returns.columns)
    assert covariance.columns.equals(asset_returns.columns)
    assert np.isfinite(covariance.to_numpy()).all()
    print(f"Fitted dates: {[str(date.date()) for date in covariances]}")
    print(f"Annual asset log-return covariance at {as_of.date()}:\n{covariance}")
    print(f"Latest covariance ({latest_date.date()}):\n{latest_covariance}")
    return spec, rolling, covariance


if __name__ == "__main__":
    run_local_test(case=1)
