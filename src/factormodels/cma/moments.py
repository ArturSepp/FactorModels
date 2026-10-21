"""Explicit log-moment convention for arithmetic capital market assumptions."""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

__all__ = ["annual_arithmetic_from_log"]


def annual_arithmetic_from_log(
    log_mean: pd.Series,
    log_variance: pd.Series,
    annualization_factor: float = 1.0,
) -> pd.Series:
    """Apply the declared second-order arithmetic-mean bridge to log moments.

    The result is ``A * (log_mean + 0.5 * log_variance)`` for ``A`` periods
    per year. This is an approximate CMA convention, not exact compounding or
    an exact expectation of exponentiated returns. It does not transform a
    covariance matrix or estimate either moment.

    Inputs must describe the same return quantity, observation period, currency
    and estimation convention. To reconcile a factor-based arithmetic forecast,
    bridge the asset and factor means separately, then subtract the loaded factor
    means. Bridging a log regression residual alone omits loading and covariance
    terms in the arithmetic intercept.

    Args:
        log_mean: Nonempty, uniquely labelled finite real log-return means.
        log_variance: Nonnegative finite log-return variances with exactly the
            same labels and ordering as ``log_mean``.
        annualization_factor: Positive finite number of periods per year.
            Use 4 for quarterly, 12 for monthly, or 1 for already annualised
            log means and variances. No frequency is inferred from the labels.

    Returns:
        pd.Series: Annual arithmetic-mean approximations retaining the input
        mean's index and name. Both input objects remain unchanged.

    Raises:
        ValueError: Inputs are invalid, misaligned, nonfinite or would produce
            nonfinite output.
    """
    if (not isinstance(annualization_factor, Real)
            or isinstance(annualization_factor, (bool, np.bool_))
            or not np.isfinite(annualization_factor) or annualization_factor <= 0):
        raise ValueError("annualization_factor must be a positive finite real scalar")
    arrays = []
    for name, value in (("log_mean", log_mean), ("log_variance", log_variance)):
        if not isinstance(value, pd.Series) or value.empty:
            raise ValueError(f"{name} must be a nonempty labelled Series")
        index = value.index
        missing_labels = (any((codes < 0).any() for codes in index.codes)
                          if isinstance(index, pd.MultiIndex) else index.hasnans)
        if not index.is_unique or missing_labels:
            raise ValueError(f"{name} labels must be unique and nonmissing")
        if (not pd.api.types.is_numeric_dtype(value.dtype)
                or pd.api.types.is_bool_dtype(value.dtype)
                or np.iscomplexobj(value)):
            raise ValueError(f"{name} must contain finite real values")
        values = value.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite real values")
        arrays.append(values)
    if not log_mean.index.equals(log_variance.index):
        raise ValueError("log_variance must match log_mean labels in the same order")
    if np.any(arrays[1] < 0):
        raise ValueError("log_variance must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(annualization_factor) * (arrays[0] + 0.5 * arrays[1])
    if not np.isfinite(result).all():
        raise ValueError("annual arithmetic approximation must be finite")
    return pd.Series(result, index=log_mean.index.copy(), name=log_mean.name)
