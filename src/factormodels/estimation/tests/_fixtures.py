"""Deterministic panels for public estimation and risk export contracts."""
import numpy as np
import pandas as pd
from factorlasso import LassoModel, LassoModelType
from factormodels.estimation import FactorCovarEstimator


def panels():
    """Return monthly and quarterly assets driven by two correlated factors."""
    rng = np.random.default_rng(2092026)
    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    x = pd.DataFrame(rng.normal(.002, .035, (108, 2)), index=dates, columns=["f", "g"])
    common = rng.normal(0., .02, len(dates))
    y = pd.DataFrame({
        "a": .7 * x.f + .1 * x.g + common,
        "b": .2 * x.f - .3 * x.g + .8 * common + rng.normal(0., .006, len(dates)),
    })
    q = pd.DataFrame({"c": (.4 * x.f + rng.normal(0., .02, len(dates))).resample("QE").sum()})
    return np.exp(x.cumsum()), {"ME": y, "QE": q}


def estimator(**kwargs):
    """Construct a portable, deterministic fit with independent cadence spans."""
    options = dict(
        lasso_model=LassoModel(
            model_type=LassoModelType.LASSO, span=36,
            span_freq_dict={"ME": 36, "QE": 12}, warmup_period=6,
            reg_lambda=1e-5, solver="CLARABEL",
        ),
        factor_returns_freq="ME", factor_covar_span=36,
        rebalancing_freq="ME",
    )
    options.update(kwargs)
    return FactorCovarEstimator(**options)
