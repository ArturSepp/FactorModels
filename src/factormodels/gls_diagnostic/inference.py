"""Purpose: Infer factor premia from supplied cross-sectional forecasts using GLS.
Inputs: Fixed loadings, residual covariance, total forecasts, cash and admitted alpha.
Outputs: Premium estimates, forecast reconciliation, conditional intervals and sensitivity boxes.
Units: Arithmetic forecast returns and residual covariance share the declared horizon.
Conditioning: Student-t intervals assume Gaussian errors with covariance tau-squared Omega.
Ownership: FM owns second-stage inference; geometry supplies rank and projections.

Cash and admitted alpha are subtracted once. No loading estimation, covariance
repair or annualisation is performed. First-stage estimation uncertainty is excluded.
Geometry follows Aitken (1935), doi:10.1017/S0370164600014346.
"""
from __future__ import annotations
from collections.abc import Sequence
from numpy.typing import ArrayLike


from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import t
from factormodels.gls_diagnostic.geometry import compute_gls_geometry
from factormodels.gls_diagnostic._validation import _scalar, _vector

__all__ = [
    'GlsPremiaResult',
    'infer_factor_premia',
]

@dataclass(frozen=True)
class GlsPremiaResult:
    """Labelled inference, exact forecast reconciliation and explicit assumptions.

    Attributes:
        premia (Series):
            Minimum-norm factor coordinates; only uniquely identified at full rank.
        assets (DataFrame):
            Total forecast, cash, credited alpha, GLS target, fitted return and residual.
        factors (DataFrame):
            Identified estimates, pointwise conditional intervals, VIF and gross weights.
        covariance (DataFrame):
            Conditional premium covariance; NaN when intervals are unavailable.
        mimicking_weights (DataFrame):
            Asset-by-factor unit-exposure weights; NaN at deficient rank.
        sensitivity_bounds (DataFrame):
            Exact marginal forecast-box bounds at full rank, separate from confidence intervals.
        summary (dict):
            Rank, scale, degrees of freedom, interval status and working assumptions.
    """

    premia: pd.Series
    assets: pd.DataFrame
    factors: pd.DataFrame
    covariance: pd.DataFrame
    mimicking_weights: pd.DataFrame
    sensitivity_bounds: pd.DataFrame
    summary: dict


def infer_factor_premia(B: ArrayLike, Omega: ArrayLike, total_returns: ArrayLike, *, cash_rate: float | np.integer | np.floating, alpha: ArrayLike | None=None, alpha_weight: float | np.integer | np.floating=0.,
                       confidence_level: float | np.integer | np.floating=.95, rcond: float | np.integer | np.floating | None=None, lower_returns: ArrayLike | None=None, upper_returns: ArrayLike | None=None,
                       cash_bounds: Sequence[float] | np.ndarray | pd.Series | None=None) -> GlsPremiaResult:
    """Infer factor premia from provider CMAs conditional on a fitted risk model.

    Args:
        B, Omega (array-like):
            Estimated loadings and residual-return covariance, in matching asset order.
        total_returns (array-like of shape (n_assets,)):
            Provider arithmetic total-return forecasts in decimal units and one currency.
        cash_rate (float):
            One common cash forecast, in the same return units and horizon convention.
        alpha (array-like of shape (n_assets,) or None, default None):
            Declared alpha vector (including any asset-specific admission weights).
            Supply historical alpha directly only if full historical admission is intended.
        alpha_weight (float, default 0):
            Diagnostic multiplier in [0, 1]. Target = total_returns - cash_rate
            - alpha_weight * alpha. Nonzero weight requires an explicit alpha vector.
        confidence_level (float, default 0.95):
            Pointwise conditional Student-t coverage under the additional working model.
        rcond (float or None, default None):
            SVD rank cutoff passed to :func:`compute_gls_geometry`.
        lower_returns, upper_returns (array-like or None, default None):
            Optional paired total-return box endpoints in the same asset order and units.
        cash_bounds (pair of float or None, default None):
            One shared cash interval for the box. None keeps cash fixed. These sensitivity
            boxes are neither factor quantiles nor statistical confidence intervals.

    Returns:
        GlsPremiaResult:
            Annual outputs when inputs are annual. No annualisation or extra sample-size
            divisor is applied. Coordinate CIs and mimicking portfolios are unavailable
            at deficient rank; zero residual scale and N <= K also suppress CIs.

    Notes:
        Forecast errors are Gaussian with covariance tau squared times Omega. B, Omega,
        cash, alpha and factor definitions are fixed. This assumption is not implied by
        a return risk model and excludes first-stage estimation uncertainty. Data fetching,
        price conversion, loading estimation and report rendering belong to the caller.
    """
    cash = _scalar(cash_rate, "cash_rate")
    weight = _scalar(alpha_weight, "alpha_weight")
    confidence = _scalar(confidence_level, "confidence_level")
    if not 0 <= weight <= 1:
        raise ValueError("alpha_weight must be in [0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if alpha is None and weight != 0:
        raise ValueError("nonzero alpha_weight requires an explicit alpha vector")
    geometry = compute_gls_geometry(B, Omega, rcond=rcond)
    n, k = geometry.beta.shape
    ids = B.index.copy() if isinstance(B, pd.DataFrame) else pd.RangeIndex(n, name="asset")
    names = B.columns.copy() if isinstance(B, pd.DataFrame) else pd.RangeIndex(k, name="factor")
    total = _vector(total_returns, B, "total_returns")
    admission = np.zeros(n) if alpha is None else _vector(alpha, B, "alpha")
    credited = weight * admission
    target = total - cash - credited
    white_target = np.linalg.solve(geometry.chol, target)
    estimate = geometry.white_pinv @ white_target
    residual = geometry.residual_projector @ target
    white_residual = np.linalg.solve(geometry.chol, residual)
    sse = float(white_residual @ white_residual)
    rank, df = geometry.rank, n - geometry.rank
    unique = rank == k
    status = ("unavailable_rank_deficient" if not unique else
              "unavailable_residual_df" if df <= 0 else
              "unavailable_zero_residual_scale"
              if sse <= max(float(white_target @ white_target), 1e-300) * 1e-24 else
              "conditional_working_model")
    covariance = np.full((k, k), np.nan)
    se, lower, upper = (np.full(k, np.nan) for _ in range(3))
    scale = critical = None
    inverse_gram = geometry.white_pinv @ geometry.white_pinv.T
    if status == "conditional_working_model":
        scale = sse / df
        critical = float(t.isf((1. - confidence) / 2., df))
        covariance = scale * inverse_gram
        se = np.sqrt(np.diag(covariance))
        lower, upper = estimate - critical * se, estimate + critical * se
        if not all(np.isfinite(x).all() for x in (covariance, se, lower, upper)):
            raise ValueError("Conditional GLS intervals overflow; check units and identification")
    q = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T)
    vif = np.sum(geometry.white**2, axis=0) * np.diag(inverse_gram)
    table = pd.DataFrame(dict(estimate=estimate if unique else np.full(k, np.nan),
                             standard_error=se, lower=lower, upper=upper, status=status,
                             variance_inflation=vif if unique else np.full(k, np.nan),
                             gross_weight=np.abs(q).sum(axis=0) if unique else np.full(k, np.nan)),
                         index=names)
    bounds = pd.DataFrame(columns=["lower", "upper", "status"], index=names)
    if (lower_returns is None) != (upper_returns is None):
        raise ValueError("lower_returns and upper_returns must be supplied together")
    if cash_bounds is not None and lower_returns is None:
        raise ValueError("cash_bounds requires return box endpoints")
    if lower_returns is not None:
        lo, hi = (_vector(v, B, name) for v, name in (
            (lower_returns, "lower_returns"), (upper_returns, "upper_returns")))
        if np.any(lo > hi):
            raise ValueError("lower_returns must not exceed upper_returns")
        cash_lo = cash_hi = cash
        if cash_bounds is not None:
            if len(cash_bounds) != 2:
                raise ValueError("cash_bounds must contain two ordered finite endpoints")
            cash_lo, cash_hi = (_scalar(x, "cash_bounds") for x in cash_bounds)
            if cash_lo > cash_hi:
                raise ValueError("cash_bounds must contain two ordered finite endpoints")
        positive, negative, funding = np.maximum(q, 0), np.minimum(q, 0), -q.sum(axis=0)
        low = positive.T @ lo + negative.T @ hi - q.T @ credited
        high = positive.T @ hi + negative.T @ lo - q.T @ credited
        low += np.minimum(funding * cash_lo, funding * cash_hi)
        high += np.maximum(funding * cash_lo, funding * cash_hi)
        bounds = pd.DataFrame(dict(lower=low if unique else np.full(k, np.nan),
                                   upper=high if unique else np.full(k, np.nan),
                                   status="sensitivity_box" if unique else status), index=names)
    assets = pd.DataFrame(dict(total_return=total, cash=cash, alpha=admission,
                               credited_alpha=credited, gls_target=target,
                               fitted=geometry.beta @ estimate, residual=residual), index=ids)
    summary = dict(method="conditional_gls_student_t", confidence_level=confidence,
                   scope="pointwise", status=status, n_assets=n, n_factors=k, rank=rank,
                   residual_df=df, weighted_residual_sse=sse, scale_squared=scale,
                   critical_value=critical, relative_cutoff=geometry.relative_cutoff,
                   alpha_weight=weight, cash_rate=cash,
                   coordinate_convention="unique" if unique else "minimum_norm_nonidentified",
                   assumptions="y = B lambda + e; e ~ N(0, tau^2 Omega). B, Omega, cash, "
                   "alpha and factor definitions are fixed. Forecast-error covariance is "
                   "assumed proportional to residual return covariance. No intercept.",
                   excluded_uncertainty="First-stage loadings, residual-risk and alpha estimation; "
                   "cash and factor definitions; forecast-error covariance misspecification.",
                   interpretation="Conditional pointwise intervals, not simultaneous coverage, "
                   "provider accuracy, adviser dispersion or realized-return prediction.")
    return GlsPremiaResult(pd.Series(estimate, index=names, name="premium"), assets, table,
                           pd.DataFrame(covariance, index=names, columns=names),
                           pd.DataFrame(q if unique else np.full_like(q, np.nan),
                                        index=ids, columns=names), bounds, summary)
