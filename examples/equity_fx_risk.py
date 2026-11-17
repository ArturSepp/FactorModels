"""Offline native local fit, structural FX augmentation and QIS active-risk comparison.

Run: python -m examples.equity_fx_risk

Synthetic monthly local log-excess responses and supplied arithmetic FX-forward
payoffs use one common grid. The example checks assembly, not observed-stock
accuracy or self-financed accounting. No provider, price index for zero-cost
payoffs, or application package is required.
"""

from enum import IntEnum

import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType

from factormodels import (
    FactorCovarEstimator,
    EquityFxRiskSpec,
    augment_equity_fx_model,
    build_currency_exposure,
)


class LocalTests(IntEnum):
    """Named offline workflows; integer values remain accepted for existing callers."""

    SYNTHETIC = 1


def run_local_test(case: LocalTests | int = LocalTests.SYNTHETIC):
    """Return a native fitted local model and augmented covariance with measurable FX dependence."""
    case = LocalTests(case)
    if case != 1:
        raise ValueError("The equity FX example supports only case 1")
    rng = np.random.default_rng(711)
    dates = pd.date_range("2015-12-31", periods=85, freq="ME")
    x = pd.DataFrame(rng.normal(0.0, [0.03, 0.015], (84, 2)), index=dates[1:], columns=["LocalMarket", "Style"])
    types = {"LocalMarket": qis.ReturnTypes.LOG, "Style": qis.ReturnTypes.DIFFERENCE}
    levels = pd.DataFrame(0.0, index=dates, columns=x.columns)
    levels.loc[dates[1:]] = x.cumsum()
    levels["LocalMarket"] = 100.0 * np.exp(levels.LocalMarket)
    phi = pd.DataFrame(rng.normal(0.0, [0.035, 0.07], (84, 2)), index=dates[1:], columns=["EUR", "BRL"])
    assets = pd.Index(["USD_A", "EUR_A", "EUR_B", "BRL_A"])
    b = np.array([[0.8, 0.3], [1.0, 0.5], [1.1, -0.4], [0.9, -0.2]])
    residual = rng.normal(0.0, 0.015, (84, 4))
    residual[:, 1] += 0.4 * phi.EUR
    residual[:, 2] += 0.2 * phi.EUR
    residual[:, 3] += 0.8 * phi.BRL
    y = pd.DataFrame(x.to_numpy() @ b.T + residual, index=x.index, columns=assets)
    estimator = FactorCovarEstimator(
        lasso_model=LassoModel(
            model_type=LassoModelType.LASSO, span=36, reg_lambda=0.0, demean=True, warmup_period=12, solver="CLARABEL"
        ),
        factor_returns_freq="ME",
        factor_covar_span=36,
        residual_type="empirical",
        residual_covar_freq="ME",
        residual_covar_span=36,
        factor_return_types=types,
    )
    cutoff = dates[-1]
    local = estimator.fit_current(levels, {"ME": y}, cutoff=cutoff, assets=assets)
    h = build_currency_exposure(
        pd.Series(["USD", "EUR", "EUR", "BRL"], index=assets), 0.0, list(phi), reference_ccy="USD", cutoff=cutoff
    )
    spec = EquityFxRiskSpec(
        "ME",
        "preserve_local",
        window=84,
        min_observations=60,
        estimator="ewma",
        span=36,
        mean_adjustment="ewma",
        cross_covariance_policy="shrink_to_psd",
    )
    result = augment_equity_fx_model(
        local,
        x,
        y,
        phi,
        h,
        spec=spec,
        factor_return_types=types,
        reference_ccy="USD",
        metadata_available_date=cutoff,
        data_available_date=cutoff,
    )
    benchmark = pd.Series(0.25, index=assets)
    portfolio = pd.Series([0.1, 0.1, 0.1, 0.7], index=assets)
    full = result.to_risk_model().compute_tre_at_date(benchmark, portfolio, cutoff)
    omitted = result.to_risk_model(include_residual_fx=False).compute_tre_at_date(benchmark, portfolio, cutoff)
    np.testing.assert_allclose(result.local_covariance, local.asset_covariance, atol=1e-12)
    pd.testing.assert_frame_equal(result.residual_covariance, local.residual_covariance)
    assert full > omitted > 0.0
    assert np.linalg.eigvalsh(result.joint_covariance).min() >= -1e-12
    print(f"Annual active risk: full={full:.2%}, residual/FX omitted={omitted:.2%}")
    print(f"Explicit cross-covariance scale: {result.diagnostics['cross_covariance_scale']:.6f}")
    return local, result


if __name__ == "__main__":
    run_local_test(case=LocalTests.SYNTHETIC)
