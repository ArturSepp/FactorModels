"""
Factor construction: turning a factor-model spec and a price panel into factor NAVs.

One factor is a vol-targeted portfolio of its basket. The basket is flattened LONG SIDE FIRST into
one signed weight vector; the unlevered return is a fixed weighted sum of its constituents, and the
leverage holds it at its target volatility:

    sigma_hat_f(t) = annualised EWMA vol of Sum_k w_fk r_k(t),   span = vol_span, weekly W-WED
    l_f(t)         = sigma*_f / sigma_hat_f(t)
    weights_f(t)   = l_f(t) · w_f,   resampled to the rebalancing anchor and held between dates

Before targeting, the first 13 valid weekly observations initialize a sample covariance.
No exposure is taken during this warm-up. The subsequent EWMA updates use only later
observations, with no full-sample seed and no double counting of the initialization sample.

The levered weights are then run through ``qis.backtest_model_portfolio``, which holds UNITS
between rebalancings — the drift adjustment that separates a backtest from a weighted average of
returns.

Units: vol targets and estimates are annualised decimals; weights are dimensionless; ``vol_span``
counts periods of ``returns_freq`` (weekly), not days. Returns used for the vol estimate are LOG
returns at ``returns_freq``.

A build may override ``rebalancing_freq`` and ``vol_span`` for the whole model at once.
Reproducing a panel requires its explicit construction settings; model-wide overrides may
differ from the individual factor defaults.

Entry points: ``build_factor_panel`` (a whole model) and ``compute_factor_nav`` (one factor).

Caller-supplied specifications and prices determine the model. Catalogue selections,
data access and estimation are separate concerns.
"""
from __future__ import annotations

# packages
import numpy as np
import pandas as pd
from typing import Mapping, Optional, Union
from numbers import Integral
# qis / project
import qis as qis
from qis.portfolio.risk.ewm_covar_risk import compute_portfolio_var_np
from factormodels.spec import (
    DEFAULT_VOL_WARMUP_PERIOD, Factor, FactorModelSpec,
)

__all__ = [
    "DEFAULT_RETURNS_FREQ", "INHERIT", "build_factor_panel", "build_factor_catalogue_panel",
    "select_factor_prices", "compute_factor_nav", "compute_risk_factor_portfolio",
    "compute_live_strategic_weights", "compute_volatility_targeted_portfolio",
]

# Weekly Wednesday: avoids Monday holidays and week-end effects.
DEFAULT_RETURNS_FREQ: str = 'W-WED'


class _Inherit:
    """Sentinel for 'take the vol target from the factor'.

    ``None`` is a meaningful value on that argument — it disables vol targeting and holds the
    strategic weights unlevered — so it cannot double as 'not supplied'.
    """
    def __repr__(self) -> str:
        return 'INHERIT'


INHERIT = _Inherit()


def build_factor_panel(spec: FactorModelSpec,
                       futures_prices: pd.DataFrame,
                       rebalancing_freq: Optional[str] = None,  # None uses each basket's own anchor
                       vol_span: Optional[int] = None,          # None uses each basket's own span
                       returns_freq: str = DEFAULT_RETURNS_FREQ
                       ) -> pd.DataFrame:
    """build every factor of a model into one panel, in the model's factor order.

    Args:
        spec: The factor model.
        futures_prices: Base price panel; must carry every column in ``spec.required_tickers()``.
        rebalancing_freq: Uniform rebalancing anchor for all factors, overriding each basket's own.
            None keeps the per-basket anchors.
        vol_span: Uniform EWMA span for all factors, overriding each basket's own. None keeps the
            per-basket spans.
        returns_freq: Frequency the vol estimate is computed at.

    Returns:
        pd.DataFrame: Factor NAVs, one column per factor named by the enum value, in enum order.

    Raises:
        ValueError: If the price panel is missing any constituent the model needs.
    """
    missing = [ticker for ticker in spec.required_tickers() if ticker not in futures_prices.columns]
    if missing:
        raise ValueError(f"{spec.name}: base price panel is missing constituents {missing!r}")

    factor_navs = []
    for member in spec.factors:
        factor_navs.append(compute_factor_nav(factor=spec.basket(member),
                                              futures_prices=futures_prices,
                                              rebalancing_freq=rebalancing_freq,
                                              vol_span=vol_span,
                                              returns_freq=returns_freq))
    return pd.concat(factor_navs, axis=1, sort=False)


def build_factor_catalogue_panel(factors: Mapping[str, Factor],
                                 futures_prices: pd.DataFrame,
                                 rebalancing_freq: Optional[str] = None,
                                 vol_span: Optional[int] = None,
                                 returns_freq: str = DEFAULT_RETURNS_FREQ,
                                 include_observed_indices: bool = False,
                                 ) -> pd.DataFrame:
    """Build an ordered factor catalogue once, independently of any regression model.

    Args:
        factors: Ordered factor-name to factor mapping supplied by the caller.
        futures_prices: Base price panel carrying the catalogue constituents.
        rebalancing_freq: Optional uniform basket leverage-rebalancing override.
        vol_span: Optional uniform volatility-estimation span override.
        returns_freq: Frequency used by the construction volatility estimate.
        include_observed_indices: Include optional raw indices. Legacy NAV catalogue builds
            exclude them; an explicitly selected model always requires all its sources.

    Returns:
        pd.DataFrame: Factor NAVs in catalogue order.

    Raises:
        ValueError: If a mapping key disagrees with its factor name or constituents are missing.
    """
    factors = {name: factor for name, factor in factors.items()
               if include_observed_indices or not factor.is_observed_index}
    mismatched = [(name, factor.name) for name, factor in factors.items()
                  if name != factor.name]
    if mismatched:
        raise ValueError(f"factor catalogue keys must equal factor names, got {mismatched!r}")
    missing = {
        name: [ticker for ticker in factor.required_tickers()
               if ticker not in futures_prices.columns]
        for name, factor in factors.items()
    }
    missing = {name: tickers for name, tickers in missing.items() if tickers}
    if missing:
        raise ValueError(f"base price panel is missing catalogue constituents {missing!r}")

    factor_navs = [
        compute_factor_nav(factor=factor,
                           futures_prices=futures_prices,
                           rebalancing_freq=rebalancing_freq,
                           vol_span=vol_span,
                           returns_freq=returns_freq)
        for factor in factors.values()
    ]
    return pd.concat(factor_navs, axis=1, sort=False).reindex(columns=list(factors))


def select_factor_prices(factor: Factor, futures_prices: pd.DataFrame) -> pd.DataFrame:
    """Select basket legs and enforce a joint inception for inseparable hedged trackers."""
    tickers = factor.signed_tickers()
    missing = [t for t in tickers if t not in futures_prices.columns]
    if missing:
        raise ValueError(f"factor {factor.name!r}: base price panel is missing constituents "
                         f"{missing!r}")
    prices = futures_prices[list(tickers)]
    if factor.require_all_constituents:
        complete = prices.notna().all(axis=1)
        if not complete.any():
            raise ValueError(f"factor {factor.name!r}: no joint constituent history")
        prices = prices.loc[complete.index[complete][0]:]
        if prices.isna().any().any():
            raise ValueError(f"factor {factor.name!r}: missing constituent prices after inception")
    return prices


def compute_factor_nav(factor: Factor,
                       futures_prices: pd.DataFrame,
                       rebalancing_freq: Optional[str] = None,
                       vol_span: Optional[int] = None,
                       vol_target: Union[float, None, _Inherit] = INHERIT,
                       returns_freq: str = DEFAULT_RETURNS_FREQ,
                       ticker: Optional[str] = None
                       ) -> pd.Series:
    """the NAV of one factor, as a vol-targeted portfolio of its flattened basket.

    Args:
        factor: The factor's sides, weights, family and holding rule.
        futures_prices: Base price panel carrying the factor's constituents.
        rebalancing_freq: Overrides ``factor.rebalancing_freq`` when not None.
        vol_span: Overrides ``factor.vol_span`` when not None.
        vol_target: ``INHERIT`` resolves against the factor and its family; a float overrides it;
            ``None`` disables vol targeting and holds the strategic weights unlevered.
        returns_freq: Frequency the vol estimate is computed at.
        ticker: Name the resulting series carries. None uses ``factor.name``.

    Returns:
        pd.Series: Factor NAV, named ``ticker`` or the factor's own name.

    Raises:
        ValueError: If the price panel is missing a constituent of the factor.
    """
    if factor.is_observed_index:
        source = futures_prices[factor.long_side_tickers[0]].copy()
        valid = source.dropna()
        if valid.empty or not np.isfinite(valid).all() or (valid <= 0).any():
            raise ValueError(f'{factor.name}: observed volatility index must be positive and finite')
        return source.rename(ticker or factor.name)
    if factor.component_factors:
        # A composite holds already-constructed factor NAVs. Applying the outer QE
        # override to the blend itself would turn daily 50/50 into drifting weights.
        futures_prices = pd.concat([
            compute_factor_nav(component, futures_prices,
                               rebalancing_freq=rebalancing_freq, vol_span=vol_span,
                               returns_freq=returns_freq)
            for component in factor.component_factors], axis=1)
        rebalancing_freq = factor.rebalancing_freq
    prices = select_factor_prices(factor, futures_prices)
    resolved_vol_target = (factor.resolved_vol_target if isinstance(vol_target, _Inherit)
                           else vol_target)
    portfolio_data = compute_risk_factor_portfolio(
        prices=prices,
        strategic_weights=np.array(factor.signed_weights()),
        rebalancing_freq=(rebalancing_freq if rebalancing_freq is not None
                          else factor.rebalancing_freq),
        portfolio_vol_target=resolved_vol_target,
        vol_span=vol_span if vol_span is not None else factor.vol_span,
        returns_freq=returns_freq,
        ticker=ticker if ticker is not None else factor.name,
        vol_warmup_period=factor.vol_warmup_period)
    nav = portfolio_data.get_portfolio_nav()
    anchor = nav.first_valid_index()
    if factor.backfill_factor is not None and futures_prices.index.min() < anchor:
        # Construct the older hedged portfolio independently. Splicing its long leg
        # first would change the newer portfolio's off-quarter initial holdings.
        older = compute_factor_nav(
            factor.backfill_factor, futures_prices.loc[:anchor],
            rebalancing_freq=rebalancing_freq, vol_span=vol_span,
            returns_freq=returns_freq)
        head_index = futures_prices.index[futures_prices.index < anchor]
        head = older.reindex(head_index)
        older_anchor = older.reindex([anchor]).iloc[0]
        if (not np.isfinite(older_anchor) or older_anchor <= 0
                or not np.isfinite(head).all() or head.le(0).any()):
            raise ValueError(f"factor {factor.name!r}: incomplete or nonpositive backfill through {anchor}")
        # qis.bfill_timeseries reconstructs the entire NAV from returns, which changes
        # newer levels by rounding. This splice must retain the newer NAV verbatim.
        head = (head / older_anchor * nav.loc[anchor]).rename(nav.name)
        nav = pd.concat([head, nav.loc[anchor:]]).reindex(futures_prices.index)
    return nav


def compute_risk_factor_portfolio(prices: pd.DataFrame,
                                  strategic_weights: np.ndarray,
                                  rebalancing_freq: str = 'QE',
                                  portfolio_vol_target: Optional[float] = None,  # None: unlevered
                                  vol_span: int = 3 * 52,          # weekly periods
                                  returns_freq: str = DEFAULT_RETURNS_FREQ,
                                  ticker: str = 'factor',
                                  vol_warmup_period: int = DEFAULT_VOL_WARMUP_PERIOD,
                                  ) -> qis.PortfolioData:
    """backtest one factor as a vol-targeted portfolio of its futures basket.

    Args:
        prices: Constituent prices, one column per instrument.
        strategic_weights: Fixed weights aligned to the columns of ``prices``.
        rebalancing_freq: Pandas calendar anchor the weights are rebalanced on.
        portfolio_vol_target: Annualised decimal vol target; None holds the strategic weights,
            still restricted to the live constituents.
        vol_span: EWMA span for the vol estimate, in periods of ``returns_freq``.
        returns_freq: Frequency the vol estimate is computed at.
        ticker: Name of the resulting portfolio.
        vol_warmup_period: Initial valid returns used for the sample covariance; ignored
            when targeting is disabled. The default is 13 weekly observations.

    Returns:
        qis.PortfolioData: The backtested factor portfolio.
    """
    if portfolio_vol_target is not None:
        risk_weights = compute_volatility_targeted_portfolio(prices=prices,
                                                             strategic_weights=strategic_weights,
                                                             rebalancing_freq=rebalancing_freq,
                                                             portfolio_vol_target=portfolio_vol_target,
                                                             vol_span=vol_span,
                                                             returns_freq=returns_freq,
                                                             vol_warmup_period=vol_warmup_period)
    else:
        # the unlevered branch holds the strategic basket, but still only over what is live
        risk_weights = compute_live_strategic_weights(prices=prices,
                                                      strategic_weights=strategic_weights)
        # A DataFrame supplies its own rebalance dates to qis; a daily frame would
        # silently rebalance an unscaled multi-leg basket every day, ignoring QE.
        schedule = qis.generate_rebalancing_indicators(
            df=prices, freq=rebalancing_freq, include_start_date=True)
        risk_weights = risk_weights.loc[schedule]
    portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                  weights=risk_weights,
                                                  ticker=ticker,
                                                  rebalancing_freq=rebalancing_freq)
    return portfolio_data


def compute_live_strategic_weights(prices: pd.DataFrame,
                                   strategic_weights: np.ndarray,
                                   ) -> pd.DataFrame:
    """the strategic basket restricted to the constituents that price, renormalised PER SIDE.

    A constituent that has not started yet is an absent holding. Without side-specific
    normalisation, a basket with a missing leg would run at reduced exposure.

    **Per side, not per row.** The obvious normaliser divides each row by its sum, which is right
    for a long-only basket that sums to one and wrong for a general long-short basket.
    A zero-net basket would divide by zero. Scaling each side to its own declared total
    preserves net and gross exposure, and reduces to the row-sum formula exactly when the
    basket is long-only and sums to one.

        w_i(t) = w_i * (W_long / W_long(t))   for w_i > 0,   W_long = sum of positive w
        w_i(t) = w_i * (W_short / W_short(t)) for w_i < 0,   W_short = |sum of negative w|

    with W_side(t) summed over the live constituents only. A side with nothing live scales to zero
    rather than to infinity, so a basket with no live long leg holds nothing instead of exploding.

    When every constituent prices, each scale is exactly 1.0 and the result is bit-identical to
    the tiled static weights wherever the panel is complete.

    Args:
        prices: Constituent prices, one column per instrument. NaN means "not live at this date".
        strategic_weights: Signed declared weights aligned to the columns of ``prices``.

    Returns:
        pd.DataFrame: Date x instrument weights, on the index of ``prices``.

    Raises:
        ValueError: If the weight vector does not match the price panel's columns.
    """
    if len(strategic_weights) != prices.shape[1]:
        raise ValueError(f"strategic_weights has {len(strategic_weights)} entries against "
                         f"{prices.shape[1]} price columns")
    declared = pd.Series(np.asarray(strategic_weights, dtype=float), index=prices.columns)
    tiled = pd.DataFrame(np.tile(declared.to_numpy(), (len(prices.index), 1)),
                         index=prices.index, columns=prices.columns)
    live = tiled.where(prices.notna(), 0.0)

    long_live, short_live = live.clip(lower=0.0), -live.clip(upper=0.0)
    scales = {}
    for side, live_side in (('long', long_live), ('short', short_live)):
        declared_total = (declared.clip(lower=0.0).sum() if side == 'long'
                          else -declared.clip(upper=0.0).sum())
        live_total = live_side.sum(axis=1)
        # a side with nothing live scales to zero: hold nothing, rather than divide by zero
        scale = pd.Series(np.where(live_total > 0.0, declared_total / live_total.replace(0.0, 1.0),
                                   0.0), index=prices.index)
        scales[side] = scale
    return (long_live.multiply(scales['long'], axis=0)
            - short_live.multiply(scales['short'], axis=0))


def compute_volatility_targeted_portfolio(prices: pd.DataFrame,
                                          strategic_weights: np.ndarray,
                                          returns_freq: str = DEFAULT_RETURNS_FREQ,
                                          vol_span: int = 3 * 52,
                                          portfolio_vol_target: float = 0.15,
                                          rebalancing_freq: str = 'ME',
                                          vol_warmup_period: int = DEFAULT_VOL_WARMUP_PERIOD,
                                          ) -> pd.DataFrame:
    """scale the strategic weights so the portfolio runs at a constant target volatility.

    Args:
        prices: Constituent prices, one column per instrument.
        strategic_weights: Fixed weights aligned to the columns of ``prices``.
        returns_freq: Frequency the log returns and the vol estimate are computed at.
        vol_span: EWMA span in periods of ``returns_freq``.
        portfolio_vol_target: Annualised decimal vol target.
        rebalancing_freq: Anchor the levered weights are resampled to.
        vol_warmup_period: Minimum valid construction returns. Their sample covariance
            initializes volatility; subsequent observations update the EWMA once each.

    Returns:
        pd.DataFrame: Levered weights, indexed by rebalancing date.
    """
    returns = qis.to_returns(prices=prices, freq=returns_freq, is_log_returns=True)
    # liveness is read at the RETURN dates, so the vol estimate and the held weights see the same
    # basket. ffill carries the last traded price into a non-trading return date; a constituent
    # that has not started yet stays NaN and drops out.
    strategic_weights = compute_live_strategic_weights(
        prices=prices.reindex(returns.index, method='ffill'),
        strategic_weights=strategic_weights)
    portfolio_vol = _compute_warmup_portfolio_vol(
        returns=returns, weights=strategic_weights, vol_span=vol_span,
        vol_warmup_period=vol_warmup_period)
    instrument_portfolio_leverages = (
        portfolio_vol_target * qis.to_finite_reciprocal(data=portfolio_vol)
    )
    risk_weights = strategic_weights.multiply(instrument_portfolio_leverages, axis=0)
    risk_weights = risk_weights.resample(rebalancing_freq).last()
    return risk_weights


def _compute_warmup_portfolio_vol(returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  vol_span: int,
                                  vol_warmup_period: int,
                                  ) -> pd.Series:
    """Seed risk from the first usable sample, then continue the existing qis recursion.

    The sample covariance (ddof=1) is known only at the last warm-up observation.
    It is published there once; the next return is the first EWMA update. Missing
    unheld legs contribute zero to the seed, as in the existing portfolio-risk
    recursion. No exposure is permitted before the sample is available.
    """
    if (isinstance(vol_warmup_period, bool)
            or not isinstance(vol_warmup_period, Integral)
            or vol_warmup_period < 2):
        raise ValueError('vol_warmup_period must be an integer of at least 2')
    portfolio_vol = pd.Series(np.nan, index=returns.index, dtype=float)
    held_weights = weights.shift(1).fillna(0.0)
    valid = (returns.notna().any(axis=1)
             & weights.abs().sum(axis=1).gt(0.0)
             & (returns.notna() | held_weights.eq(0.0)).all(axis=1))
    usable = np.flatnonzero(valid.to_numpy())
    if len(usable) < vol_warmup_period:
        return portfolio_vol

    sample_positions = usable[:vol_warmup_period]
    start = int(sample_positions[-1])
    seed = returns.iloc[sample_positions].fillna(0.0).cov().to_numpy()
    annualization = qis.infer_annualisation_factor_from_df(returns)
    portfolio_vol.iloc[start] = np.sqrt(annualization) * np.sum(
        qis.compute_portfolio_risk_contributions(
            w=held_weights.iloc[start].to_numpy(), covar=seed))
    if start + 1 < len(returns):
        variance = compute_portfolio_var_np(
            returns=returns.iloc[start + 1:].fillna(0.0).to_numpy(),
            weights=held_weights.iloc[start + 1:].to_numpy(),
            span=vol_span,
            covar0=seed)
        portfolio_vol.iloc[start + 1:] = np.sqrt(annualization * np.maximum(variance, 0.0))
    return portfolio_vol
