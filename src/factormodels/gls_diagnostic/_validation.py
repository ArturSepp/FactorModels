"""Purpose: Validate labelled GLS inputs and prepare Cholesky-whitened loadings.
Inputs: Supplied arrays or numeric pandas objects with exact, unique, nonmissing axes.
Outputs: Validated arrays, labels and full whitened loadings without rank truncation.
Units: Values retain their supplied return and covariance coordinates and horizon.
Conditioning: Inputs are fixed; covariance must be SPD without repair or clipping.
Ownership: FM validation sits below geometry and accounting and imports neither.

Lists/ndarrays retain float conversion. Labelled numerical inputs require numeric
dtypes; descriptive tables explicitly checked with finite=False retain object data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

__all__ = [
]

_AXIS_ERROR = "label/order mismatch or duplicate/missing labels"
_ALIGN_ERROR = "label/order mismatch or duplicate labels"


def _numeric_labelled(value: Any, name: str) -> None:
    """Reject implicit string/object conversion for labelled numerical inputs."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        dtypes = value.dtypes if isinstance(value, pd.DataFrame) else [value.dtype]
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
            raise ValueError(f"{name} must contain finite real values")


def _array(value: ArrayLike, name: str, ndim: int) -> np.ndarray:
    """Require a nonempty finite real array without changing its order."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    _numeric_labelled(value, name)
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def _axis(value: ArrayLike, labels: pd.Index | None = None, axis: int = 0,
          *, error: str = _AXIS_ERROR) -> None:
    """Reject duplicate/missing/reordered axes; retain caller-specific error text."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        actual = value.columns if axis else value.index
        if actual.has_duplicates or actual.hasnans or (
                labels is not None and not actual.equals(labels)):
            raise ValueError(error)


def _align(value: ArrayLike, labels: pd.Index | None, axis: int = 0) -> None:
    """Apply the common axis check with accounting's established error text."""
    _axis(value, labels, axis, error=_ALIGN_ERROR)


def _vector(value: ArrayLike, loadings: ArrayLike, name: str,
            *, label_error: str = _AXIS_ERROR) -> np.ndarray:
    """Validate an asset vector against the loading rows."""
    _axis(value, loadings.index if isinstance(loadings, pd.DataFrame) else None,
          error=label_error)
    result = _array(value, name, 1)
    if len(result) != len(loadings):
        raise ValueError(f"{name} asset dimension does not match B")
    return result


def _scalar(value: float, name: str) -> float:
    """Require a finite real scalar excluding booleans."""
    if (not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_)) or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _covariance_and_cholesky(value: ArrayLike, size: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate SPD covariance once and retain its lower Cholesky factor."""
    result = _array(value, name, 2)
    if result.shape != (size, size) or not np.allclose(result, result.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric {size} by {size} matrix")
    try:
        chol = np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return result, chol


def _covariance(value: ArrayLike, size: int, name: str) -> np.ndarray:
    """Validate an SPD covariance without clipping or regularizing it."""
    return _covariance_and_cholesky(value, size, name)[0]


def _whiten_loadings(B: ArrayLike, Omega: ArrayLike, *, label_error: str = _AXIS_ERROR
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare full loadings and residual covariance without SVD or rank policy."""
    _axis(B, error=label_error)
    if isinstance(B, pd.DataFrame):
        _axis(B, axis=1, error=label_error)
    assets = B.index if isinstance(B, pd.DataFrame) else None
    _axis(Omega, assets, error=label_error)
    if isinstance(Omega, pd.DataFrame):
        _axis(Omega, Omega.index if assets is None else assets, axis=1, error=label_error)
    beta = _array(B, "B", 2)
    if np.iscomplexobj(Omega):
        raise ValueError("Omega must be real")
    _numeric_labelled(Omega, "Omega")
    residual = np.asarray(Omega, dtype=float)
    if residual.ndim == 1:
        residual = np.diag(_array(Omega, "Omega", 1))
    residual, chol = _covariance_and_cholesky(residual, len(beta), "Omega")
    white = np.linalg.solve(chol, beta)
    return beta, residual, chol, white


def _factors(lam: ArrayLike, Sig_F: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate premia and factor covariance, including labelled axes."""
    labels = lam.index if isinstance(lam, pd.Series) else None
    _align(lam, labels)
    _align(Sig_F, labels)
    _align(Sig_F, Sig_F.index if isinstance(Sig_F, pd.DataFrame) else labels, axis=1)
    premia = _array(lam, "lam", 1)
    return premia, _covariance(Sig_F, len(premia), "Sig_F")


def _labelled(value: pd.Series | pd.DataFrame, kind: type[pd.Series] | type[pd.DataFrame],
              labels: pd.Index, name: str, *, finite: bool = True) -> pd.Series | pd.DataFrame:
    """Validate an exactly aligned labelled vector or table without coercing axes."""
    if not isinstance(value, kind):
        raise ValueError(f'{name} must be a labelled {kind.__name__}')
    if value.index.has_duplicates or value.index.hasnans or not value.index.equals(labels):
        raise ValueError(f'{name} must match the exact unique index')
    if isinstance(value, pd.DataFrame) and (
            value.columns.has_duplicates or value.columns.hasnans):
        raise ValueError(f'{name} columns must be unique and nonmissing')
    dtypes = value.dtypes if isinstance(value, pd.DataFrame) else [value.dtype]
    if finite and (not all(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes)
                   or np.iscomplexobj(value)
                   or not np.isfinite(value.to_numpy(dtype=float)).all()):
        raise ValueError(f'{name} must contain finite real values')
    return value


def _labels(B: ArrayLike) -> tuple[pd.Index, pd.Index]:
    """Return the validated loading matrix's asset and factor labels."""
    n, k = np.shape(B)
    return ((B.index.copy(), B.columns.copy()) if isinstance(B, pd.DataFrame) else
            (pd.RangeIndex(n, name="asset"), pd.RangeIndex(k, name="factor")))


def _factor_vector(value: ArrayLike, labels: pd.Index, name: str) -> np.ndarray:
    """Validate factor coordinates without silently relabelling their order."""
    _axis(value, labels)
    result = _array(value, name, 1)
    if len(result) != len(labels):
        raise ValueError(f"{name} factor dimension does not match B")
    return result


def _text(value: str, name: str) -> str:
    """Require a nonempty explanatory label or provenance string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value
