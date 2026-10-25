"""QIS-backed factor return and covariance preparation.

The OP EWMA estimator remains in OP. QIS has no single exported helper with
this exact sampling, double-drop, annualisation and normalized-kernel contract,
so FM owns this small orchestration of its existing kernels.

Purpose: Prepare factor returns and factor covariance using existing QIS kernels.
Inputs: Dated factor prices, return-coordinate mapping, sampling frequency and EWMA settings.
Outputs: compute_returns_from_prices and estimate_factor_covariance return sampled returns and F.
Units: LOG factors use fractional log returns; DIFFERENCE factors keep point units and covariance uses their products.
Conditioning: Preserve sampling, double-drop, demeaning, normalization and optional annualisation order.
Ownership: FM owns the orchestration contract; QIS owns conversion and covariance calculations.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import qis as qis
from qis import to_factor_returns


__all__ = []


def compute_returns_from_prices(prices: pd.DataFrame,
                                returns_freq: Optional[str] = 'ME',
                                demean: bool = True,
                                drop_first: bool = True,
                                is_first_zero: bool = False,
                                is_log_returns: bool = True,
                                span: Optional[int] = 52,
                                factor_return_types: Optional[dict] = None
                                ) -> pd.DataFrame:
    """
    compute returns for covar matrix estimation
    """
    convert = qis.to_returns if factor_return_types is None else to_factor_returns
    options = ({"is_log_returns": is_log_returns} if factor_return_types is None else
               {"return_types": factor_return_types})
    returns = convert(prices=prices,
                             **options,
                             is_first_zero=is_first_zero,
                             drop_first=drop_first,
                             freq=returns_freq)
    if demean:
        returns = returns - qis.compute_ewm(returns, span=span)
        # returns.iloc[0, :] will be zero so shift the period
        if drop_first:
            returns = returns.iloc[1:, :]
    return returns

def estimate_factor_covariance(prices: pd.DataFrame,
                                returns_freq: str = 'W-WED',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                apply_an_factor: bool = True,
                                factor_return_types: Optional[dict] = None,
                                ) -> pd.DataFrame:
    """Estimate the final QIS EWMA covariance from the supplied price history.

    The helper samples log returns, drops the first price-difference row and,
    when demeaning, subtracts the contemporaneous EWMA mean and drops the first
    zero deviation. The ordinary covariance recursion starts at zero.
    NanBackfill.ZERO_FILL resets non-finite covariance updates to zero; it is
    not equivalent to replacing every missing input return with zero.

    This is a full-panel current fit. Slice prices first for a historical cutoff.
    Volatility normalization uses the QIS kernel that, since qis 5.31.0, seeds each
    volatility with its column's first finite squared return, not a full-array statistic.

    Args:
        prices: Date-by-ticker price panel with an ordered DatetimeIndex. Prices
            should represent the intended total-return convention.
        returns_freq: Frequency for sampling log returns, default 'W-WED'.
        span: EWMA span in sampled return observations. Decay is
            1 - 2 / (span + 1); span is not a half-life or a fixed window length.
        is_apply_vol_normalised_returns: Select the QIS normalized-return kernel,
            which reconstructs covariance using EWMA volatilities. This does not
            shrink covariance toward an identity matrix.
        demean: Subtract the current EWMA mean before covariance estimation.
            False retains raw log returns after the initial difference.
        apply_an_factor: Multiply by the annualization factor inferred from the
            sampled return index. False leaves per-observation covariance units.
        factor_return_types: Optional exact per-column QIS return types; None retains log returns.

    Returns:
        Square covariance DataFrame on the price tickers, using the last sampled
        return state. Units are fractional log-return squared, annualized when
        apply_an_factor is True.
    """
    returns = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=demean,
                                          span=span, factor_return_types=factor_return_types)
    x = returns.to_numpy()
    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(
            a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = qis.compute_ewm_covar_tensor(
            a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)

    covar_t = covar_tensor_txy[-1]
    if apply_an_factor:
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
    else:
        an_factor = 1.0
    current_covar = pd.DataFrame(an_factor * covar_t, columns=returns.columns,
                                 index=returns.columns)
    return current_covar
