"""Purpose: Construct the canonical rank-revealing GLS geometry.
Inputs: Supplied loadings B and positive-definite residual covariance Omega.
Outputs: Whitened loadings, SVD rank, projector and mimicking portfolios.
Units: Loadings and covariance retain caller-supplied compatible return coordinates.
Conditioning: B and Omega are fixed; one eps-times-dimension SVD cutoff applies.
Ownership: FM owns geometry; shared validation performs Cholesky whitening below it.
"""
from __future__ import annotations
from numpy.typing import ArrayLike


from dataclasses import dataclass
import numpy as np
from factormodels.gls_diagnostic._validation import _scalar, _vector, _whiten_loadings

__all__ = [
    'GlsGeometry',
    'compute_gls_geometry',
    'factor_mimicking_portfolios',
    'gls_projector',
    'gls_split',
]

@dataclass(frozen=True)
class GlsGeometry:
    """Validated GLS geometry with one numerical rank convention.

    Attributes:
        beta, residual, chol, white (ndarray):
            Loadings, residual covariance, its lower Cholesky factor and whitened loadings.
        singular_values (ndarray):
            Singular values of the whitened loadings (not of their Gram matrix).
        rank (int):
            Number of singular values strictly above ``cutoff``.
        relative_cutoff, cutoff (float):
            Relative and absolute singular-value cutoffs.
        white_pinv, residual_projector (ndarray):
            Whitened pseudoinverse and asset-space residual operator, respectively.
    """

    beta: np.ndarray
    residual: np.ndarray
    chol: np.ndarray
    white: np.ndarray
    singular_values: np.ndarray
    rank: int
    relative_cutoff: float
    cutoff: float
    white_pinv: np.ndarray
    residual_projector: np.ndarray


def compute_gls_geometry(B: ArrayLike, Omega: ArrayLike, *, rcond: float | np.integer | np.floating | None=None) -> GlsGeometry:
    """Build whitened GLS geometry without squaring the condition number.

    Args:
        B (array-like of shape (n_assets, n_factors)):
            Factor loadings. Pandas axes must be unique, nonmissing and exactly aligned.
        Omega (array-like of shape (n_assets,) or (n_assets, n_assets)):
            Positive residual variances or a positive-definite residual covariance.
        rcond (float or None, default None):
            Relative SVD cutoff in [0, 1). None uses eps * max(n_assets, n_factors).

    Returns:
        GlsGeometry:
            A rank-revealing geometry. Truncating a nonzero direction changes the
            effective model span and can invalidate identities under the original model.
    """
    beta, residual, chol, white = _whiten_loadings(B, Omega)
    n = len(beta)
    relative = np.finfo(float).eps * max(white.shape) if rcond is None else _scalar(
        rcond, "rcond")
    if not 0 <= relative < 1:
        raise ValueError("rcond must be a finite scalar in [0, 1)")
    u, singular, vt = np.linalg.svd(white, full_matrices=False)
    cutoff = float(relative * singular[0])
    keep = singular > cutoff
    basis = u[:, keep]
    inverse = (vt[keep].T / singular[keep]) @ basis.T
    projector = chol @ (np.eye(n) - basis @ basis.T) @ np.linalg.solve(chol, np.eye(n))
    return GlsGeometry(beta, residual, chol, white, singular, int(keep.sum()),
                       float(relative), cutoff, inverse, projector)


def gls_projector(B: ArrayLike, Omega: ArrayLike) -> np.ndarray:
    """Return the asset-space residual operator using the default SVD rank policy.

    Args:
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        ndarray:
            Residual operator of shape (n_assets, n_assets).
    """
    return compute_gls_geometry(B, Omega).residual_projector


def gls_split(m: ArrayLike, B: ArrayLike, Omega: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split excess forecasts into minimum-norm premia and orthogonal residuals.

    Args:
        m (array-like of shape (n_assets,)):
            Excess forecasts, with any admitted alpha already subtracted.
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        premia, residual : ndarray:
            Factor coordinates and asset residual. Coordinates are not uniquely
            identified at deficient rank; use :func:`infer_factor_premia` for flags.
    """
    geometry = compute_gls_geometry(B, Omega)
    mean = _vector(m, B, "m")
    premia = geometry.white_pinv @ np.linalg.solve(geometry.chol, mean)
    return premia, geometry.residual_projector @ mean


def factor_mimicking_portfolios(B: ArrayLike, Omega: ArrayLike) -> np.ndarray:
    """Return minimum-residual-variance portfolios with unit factor exposures.

    Args:
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        ndarray:
            Asset-by-factor risky weights Q, with B.T @ Q = I. Weights may be short
            or leveraged and need not sum to one; cash completes their financing.

    Raises:
        ValueError:
            If the loading matrix lacks full column rank.
    """
    geometry = compute_gls_geometry(B, Omega)
    if geometry.rank != geometry.beta.shape[1]:
        raise ValueError("unit factor exposures are unattainable: B lacks full column rank")
    return np.linalg.solve(geometry.chol.T, geometry.white_pinv.T)
