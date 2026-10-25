"""Annualized excess-Sharpe accounting for any linear factor model.

MATF is the motivating factor set; these functions impose no particular factor names.
Inputs must share units, factor order and asset order. Residual covariance is positive
definite, supplied as a full matrix or a diagonal variance vector. Returned arrays are
in input order. No forecasts, histories, annualization or covariance repairs are inferred.

The functions implement the achievable-Sharpe and GLS identities of Sepp and Kastenholz
(2026). qis.RiskModel owns risk contributions and tracking-error reporting; the closed-form
access, forecast and fixed-book completion identities here are absent from its API.

Purpose: Evaluate closed-form factor access, overlay and fixed-book completion identities.
Inputs: Aligned factor premia, factor covariance, loadings, residual covariance and optional forecasts/holdings.
Outputs: Accounting functions return attainable excess-Sharpe capacities, decompositions and completion status.
Units: Means and covariances share one horizon; squared Sharpe is annualised when inputs are annual.
Conditioning: Sigma = B F B.T + Omega with SPD risk; full whitened loadings are retained without rank truncation in capacity formulas.
Ownership: FM owns these algebraic identities; QIS owns portfolio risk contributions and tracking-error reporting.
"""

from __future__ import annotations
from numpy.typing import ArrayLike


from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from factormodels.gls_diagnostic._validation import (
    _array, _align, _covariance, _factors, _vector, _whiten_loadings, _ALIGN_ERROR,
)
from factormodels.gls_diagnostic.geometry import compute_gls_geometry












__all__ = ['factor_ceiling', 'achievable_sharpe2', 'fpir', 'holdings_decomposition', 'hurdle_premia', 'ceiling_premium_correlation_split', 'fixed_exposure_fpir', 'tangency_access_split', 'overlay_value', 'partial_overlay_value', 'CompletionStatus', 'CompletionResult', 'completion_sharpe2', 'ir_bound']


def _model(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate all factor-model inputs without changing their order."""
    premia, factor = _factors(lam, Sig_F)
    beta, residual, chol, white = _whiten_loadings(B, Omega, label_error=_ALIGN_ERROR)
    labels = B.columns if isinstance(B, pd.DataFrame) else None
    _align(lam, labels)
    _align(Sig_F, labels)
    _align(Sig_F, labels, axis=1) if isinstance(Sig_F, pd.DataFrame) else None
    if beta.shape[1] != len(premia):
        raise ValueError("B factor dimension does not match lam")
    return premia, factor, beta, residual, chol, white




def factor_ceiling(lam: ArrayLike, Sig_F: ArrayLike) -> float:
    """Return the annualized squared excess Sharpe with unrestricted factor access."""
    premia, factor = _factors(lam, Sig_F)
    return float(premia @ np.linalg.solve(factor, premia))


def achievable_sharpe2(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> float:
    """Return systematic squared Sharpe in the asset universe, including deficient rank."""
    premia, factor, _, _, _, white = _model(lam, Sig_F, B, Omega)
    precision = np.linalg.solve(factor, np.eye(len(premia)))
    z = precision @ premia
    return float(premia @ z - z @ np.linalg.solve(precision + white.T @ white, z))


def fpir(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> float:
    """Return achievable squared Sharpe divided by the nonzero factor ceiling."""
    ceiling = factor_ceiling(lam, Sig_F)
    if ceiling == 0:
        raise ValueError("FPIR is undefined for zero factor premia")
    return achievable_sharpe2(lam, Sig_F, B, Omega) / ceiling








def holdings_decomposition(w: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return factor-replicating holdings, factor-neutral holdings and book exposures."""
    geometry = compute_gls_geometry(B, Omega)
    weights = _vector(w, B, 'w', label_error=_ALIGN_ERROR)
    exposure = geometry.beta.T @ weights
    neutral_weights = geometry.residual_projector.T @ weights
    return weights - neutral_weights, neutral_weights, exposure


def hurdle_premia(lam: ArrayLike, Sig_F: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each hurdle premium, conditional variance and drop-one squared-Sharpe gain."""
    premia, factor = _factors(lam, Sig_F)
    precision = np.linalg.solve(factor, np.eye(len(premia)))
    variance = 1.0 / np.diag(precision)
    z = precision @ premia
    return premia - variance * z, variance, variance * z**2


def ceiling_premium_correlation_split(lam: ArrayLike, Sig_F: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return additive standalone-premium and correlation contributions by factor."""
    premia, factor = _factors(lam, Sig_F)
    standalone = premia**2 / np.diag(factor)
    return standalone, premia * np.linalg.solve(factor, premia) - standalone




def fixed_exposure_fpir(e: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> float:
    """Return systematic-risk share for an attainable nonzero factor exposure target."""
    geometry = compute_gls_geometry(B, Omega)
    beta, residual = geometry.beta, geometry.residual
    labels = B.columns if isinstance(B, pd.DataFrame) else None
    _align(e, labels)
    _align(Sig_F, labels)
    _align(Sig_F, labels, axis=1) if isinstance(Sig_F, pd.DataFrame) else None
    exposure = _array(e, "e", 1)
    factor = _covariance(Sig_F, beta.shape[1], "Sig_F")
    if len(exposure) != beta.shape[1]:
        raise ValueError("e factor dimension does not match B")
    weights = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T) @ exposure
    if not np.allclose(beta.T @ weights, exposure, rtol=1e-10, atol=1e-12):
        raise ValueError("target exposure is unattainable")
    systematic = float(exposure @ factor @ exposure)
    if systematic == 0:
        raise ValueError("FPIR is undefined for zero exposure")
    return systematic / (systematic + float(weights @ residual @ weights))


def tangency_access_split(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return per-factor tangency contributions split into factor and access risk."""
    premia, factor, beta, residual, _, _ = _model(lam, Sig_F, B, Omega)
    geometry = compute_gls_geometry(B, Omega)
    covariance = beta @ factor @ beta.T + residual
    exposure = beta.T @ np.linalg.solve(covariance, beta @ premia)
    mimicking = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T)
    access = mimicking.T @ residual @ mimicking
    return exposure * (factor @ exposure), exposure * (access @ exposure)


def overlay_value(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike, a: ArrayLike) -> tuple[float, float]:
    """Return full-factor overlay gain and augmented capacity for m = B lam + a."""
    premia, factor, beta, residual, _, white = _model(lam, Sig_F, B, Omega)
    alpha = _vector(a, B, 'a', label_error=_ALIGN_ERROR)
    z = np.linalg.solve(factor, premia)
    g = beta.T @ np.linalg.solve(residual, alpha)
    h = np.linalg.solve(factor, np.eye(len(premia))) + white.T @ white
    gain = float((z - g) @ np.linalg.solve(h, z - g))
    capacity = float(premia @ z + alpha @ np.linalg.solve(residual, alpha))
    return gain, capacity


def _hedge_indices(hedgeable: ArrayLike | None, size: int) -> np.ndarray:
    """Resolve unique factor indices or a boolean mask; None means all factors."""
    if hedgeable is None:
        return np.arange(size)
    values = np.asarray(hedgeable)
    if values.ndim != 1:
        raise ValueError("hedgeable must be a one-dimensional index sequence or mask")
    if values.dtype == bool:
        if len(values) != size:
            raise ValueError("hedgeable mask has the wrong length")
        return np.flatnonzero(values)
    if values.size == 0:
        return np.array([], dtype=int)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("hedgeable indices must be integers")
    if np.any(values < 0) or np.any(values >= size) or len(np.unique(values)) != len(values):
        raise ValueError("hedgeable indices are duplicated or outside the factor axis")
    return values.astype(int)


def partial_overlay_value(lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike, a: ArrayLike, hedgeable: ArrayLike | None) -> tuple[float, float]:
    """Return overlay gain and augmented capacity when asset weights may be reoptimized."""
    premia, factor, beta, residual, _, _ = _model(lam, Sig_F, B, Omega)
    mean = beta @ premia + _vector(a, B, 'a', label_error=_ALIGN_ERROR)
    covariance = beta @ factor @ beta.T + residual
    baseline = float(mean @ np.linalg.solve(covariance, mean))
    selected = _hedge_indices(hedgeable, len(premia))
    if not len(selected):
        return 0.0, baseline
    hedge = factor[np.ix_(selected, selected)]
    cross = beta @ factor[:, selected]
    adjusted_mean = mean - cross @ np.linalg.solve(hedge, premia[selected])
    adjusted_covar = covariance - cross @ np.linalg.solve(hedge, cross.T)
    capacity = float(premia[selected] @ np.linalg.solve(hedge, premia[selected])
                     + adjusted_mean @ np.linalg.solve(adjusted_covar, adjusted_mean))
    return capacity - baseline, capacity


class CompletionStatus(str, Enum):
    """Whether fixed-book completion has attainable finite weights or only a bound."""

    FINITE = "finite"
    SUPREMUM = "supremum"
    FIXED_BOOK = "fixed_book"
    NO_POSITIVE_MEAN = "no_positive_mean"
    ZERO_VARIANCE = "zero_variance"


@dataclass(frozen=True)
class CompletionResult:
    """Signed-Sharpe bound, attainment status and optional optimal exposure/overlay.

    ``sharpe2`` is a positive-Sharpe squared bound, not the square of a negative
    book Sharpe; it is None when no positive mean is attainable. ``signed_sharpe``
    may be an unattained supremum. ``hedge_only_sharpe`` refers to cancelling the
    selected factor exposures without premium-seeking leverage. Zero/zero is None.
    """

    signed_sharpe: float | None
    sharpe2: float | None
    status: CompletionStatus
    exposures: np.ndarray | None
    overlays: np.ndarray | None
    hedge_only_sharpe: float | None


def _signed(mean: float | np.integer | np.floating, variance: float | np.integer | np.floating) -> float | None:
    """Return signed Sharpe, infinity for arbitrage and None for zero/zero."""
    if variance == 0:
        return None if mean == 0 else float(np.copysign(np.inf, mean))
    return float(mean / np.sqrt(variance))


def completion_sharpe2(w: ArrayLike, m: ArrayLike, lam: ArrayLike, Sig_F: ArrayLike, B: ArrayLike, Omega: ArrayLike, hedgeable: ArrayLike | None=None) -> CompletionResult:
    """Maximize positive signed Sharpe with the asset book fixed and selected overlays free.

    With positive residual mean and variance the optimum is finite. Nonpositive
    conditional mean and nonzero hedge premia yield an unattained factor-direction
    supremum. Inputs are SPD; a zero book is handled as a separate degenerate case.
    No leverage, funding, margin or trading-cost constraint is inferred.
    """
    premia, factor, beta, residual, _, _ = _model(lam, Sig_F, B, Omega)
    weights, mean = _vector(w, B, 'w', label_error=_ALIGN_ERROR), _vector(m, B, 'm', label_error=_ALIGN_ERROR)
    exposure = beta.T @ weights
    c = float(weights @ (mean - beta @ premia))
    v = float(weights @ residual @ weights)
    selected = _hedge_indices(hedgeable, len(premia))
    hedged = exposure.copy()
    hedged[selected] = 0
    hedge_sharpe = _signed(c + float(hedged @ premia), v + float(hedged @ factor @ hedged))
    if not len(selected):
        signed = _signed(float(weights @ mean), v + float(exposure @ factor @ exposure))
        positive = signed**2 if signed is not None and signed > 0 else None
        return CompletionResult(signed, positive, CompletionStatus.FIXED_BOOK,
                                exposure, np.zeros_like(exposure), hedge_sharpe)
    other = np.setdiff1d(np.arange(len(premia)), selected)
    hedge = factor[np.ix_(selected, selected)]
    cross = factor[np.ix_(selected, other)]
    shift = np.linalg.solve(hedge, cross @ exposure[other])
    conditional_mean = c + float(exposure[other] @ (
        premia[other] - cross.T @ np.linalg.solve(hedge, premia[selected])))
    conditional_var = v + float(exposure[other] @ (
        factor[np.ix_(other, other)] - cross.T @ np.linalg.solve(hedge, cross)) @ exposure[other])
    if conditional_var < 0:
        raise ValueError("negative conditional variance; covariance is numerically unstable")
    direction = np.linalg.solve(hedge, premia[selected])
    ceiling = float(premia[selected] @ direction)
    if conditional_var == 0:
        if conditional_mean != 0:
            raise ValueError("zero conditional variance with nonzero mean is unsupported")
        if ceiling == 0:
            return CompletionResult(None, None, CompletionStatus.ZERO_VARIANCE,
                                    None, None, hedge_sharpe)
        total = exposure.copy()
        total[selected] = direction - shift
        return CompletionResult(np.sqrt(ceiling), ceiling, CompletionStatus.FINITE,
                                total, total - exposure, hedge_sharpe)
    if conditional_mean <= 0:
        if ceiling == 0:
            return CompletionResult(0.0, None, CompletionStatus.NO_POSITIVE_MEAN,
                                    None, None, hedge_sharpe)
        return CompletionResult(np.sqrt(ceiling), ceiling, CompletionStatus.SUPREMUM,
                                None, None, hedge_sharpe)
    optimum = ceiling + conditional_mean**2 / conditional_var
    total = exposure.copy()
    total[selected] = (conditional_var / conditional_mean) * direction - shift
    return CompletionResult(np.sqrt(optimum), optimum, CompletionStatus.FINITE,
                            total, total - exposure, hedge_sharpe)


def ir_bound(m: ArrayLike, Sigma: ArrayLike) -> float:
    """Return squared excess-return capacity of zero-net-investment active holdings."""
    mean, covariance = _factors(m, Sigma)
    precision_mean = np.linalg.solve(covariance, mean)
    precision_one = np.linalg.solve(covariance, np.ones(len(mean)))
    return float(mean @ precision_mean - precision_mean.sum()**2 / precision_one.sum())
