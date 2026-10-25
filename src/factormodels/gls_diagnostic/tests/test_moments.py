"""Independent arithmetic-moment references and labelled-input contracts."""
import numpy as np
import pandas as pd
import pytest

import factormodels as tf
from factormodels.gls_diagnostic import annual_arithmetic_from_log


@pytest.mark.parametrize("annualization, expected", [
    (4., [0.10, -0.03]), (12., [0.30, -0.09]), (1., [0.025, -0.0075]),
])
def test_period_reference_and_public_exports(annualization, expected):
    """Explicit quarterly, monthly and annual values fix the half-variance rule."""
    mean = pd.Series([0.02, -0.01], index=pd.Index(["equity", "rates"], name="factor"), name="mean")
    variance = pd.Series([0.01, 0.005], index=mean.index, name="variance")
    original_mean, original_variance = mean.copy(deep=True), variance.copy(deep=True)
    assert tf.annual_arithmetic_from_log is annual_arithmetic_from_log
    assert "annual_arithmetic_from_log" in tf.__all__
    result = tf.annual_arithmetic_from_log(mean, variance, annualization)
    pd.testing.assert_series_equal(result, pd.Series(expected, index=mean.index, name=mean.name))
    pd.testing.assert_series_equal(mean, original_mean)
    pd.testing.assert_series_equal(variance, original_variance)
    result.iloc[0] = 9.
    pd.testing.assert_series_equal(mean, original_mean)
    pd.testing.assert_series_equal(variance, original_variance)


def test_complete_intercept_bridge_includes_leverage():
    """A twice-levered factor needs an arithmetic intercept even with no residual."""
    factor_mean = pd.Series([0.01], index=["factor"])
    factor_variance = pd.Series([0.01], index=factor_mean.index)
    asset_mean = pd.Series([0.02], index=["asset"])
    asset_variance = pd.Series([0.04], index=asset_mean.index)
    factor_arithmetic = annual_arithmetic_from_log(factor_mean, factor_variance, 4.)
    asset_arithmetic = annual_arithmetic_from_log(asset_mean, asset_variance, 4.)
    intercept = asset_arithmetic.iloc[0] - 2 * factor_arithmetic.iloc[0]
    assert asset_arithmetic.iloc[0] == pytest.approx(0.16)
    assert factor_arithmetic.iloc[0] == pytest.approx(0.06)
    assert intercept == pytest.approx(0.04)
    zero_residual = pd.Series([0.], index=asset_mean.index)
    residual_only = annual_arithmetic_from_log(zero_residual, zero_residual, 4.)
    assert residual_only.iloc[0] == 0.
    assert not np.isclose(intercept, residual_only.iloc[0])
    assert 2 * factor_arithmetic.iloc[0] + intercept == pytest.approx(asset_arithmetic.iloc[0])


@pytest.mark.parametrize("annualization", [0., -1., np.nan, np.inf, True, np.bool_(False), "12", [12], 1+0j])
def test_invalid_annualization_is_rejected(annualization):
    """No implicit frequency coercion, zero scale or nonfinite multiplier is accepted."""
    mean = pd.Series([0.02], index=["asset"])
    with pytest.raises(ValueError, match="annualization_factor"):
        annual_arithmetic_from_log(mean, mean, annualization)


@pytest.mark.parametrize("invalid", [
    [0.01, 0.02], pd.Series(dtype=float), pd.Series([0.01, 0.02], index=["a", "a"]),
    pd.Series([0.01, 0.02], index=["a", None]), pd.Series([np.nan, 0.02], index=["a", "b"]),
    pd.Series([np.inf, 0.02], index=["a", "b"]), pd.Series([0.01+0j, 0.02], index=["a", "b"]),
    pd.Series(["0.01", "0.02"], index=["a", "b"]), pd.Series([True, False], index=["a", "b"]),
    pd.Series([0.01, pd.NA], index=["a", "b"], dtype="Float64"),
])
@pytest.mark.parametrize("field", ["log_mean", "log_variance"])
def test_invalid_moments_are_rejected(invalid, field):
    """Both moment vectors must be finite real Series with meaningful unique labels."""
    inputs = {"log_mean": pd.Series([0.02, -0.01], index=["a", "b"]),
              "log_variance": pd.Series([0.01, 0.005], index=["a", "b"])}
    inputs[field] = invalid
    with pytest.raises(ValueError, match=field):
        annual_arithmetic_from_log(**inputs)


@pytest.mark.parametrize("variance", [
    pd.Series([0.01, 0.005], index=["b", "a"]),
    pd.Series([0.01], index=["a"]),
    pd.Series([0.01, -0.005], index=["a", "b"]),
])
def test_misaligned_or_negative_variances_are_rejected(variance):
    """Labels cannot align implicitly, and a negative variance cannot be uplifted."""
    mean = pd.Series([0.02, -0.01], index=["a", "b"])
    with pytest.raises(ValueError, match="log_variance"):
        annual_arithmetic_from_log(mean, variance)


def test_multiindex_zero_variance_and_default_annual_units():
    """Complete MultiIndex labels survive, and zero variance gives the annual log mean."""
    labels = pd.MultiIndex.from_tuples([("asset", "USD"), ("factor", "EUR")], names=["kind", "ccy"])
    mean = pd.Series([0.02, -0.01], index=labels, name="expected_return")
    variance = pd.Series([0., 0.], index=labels)
    pd.testing.assert_series_equal(annual_arithmetic_from_log(mean, variance), mean)
    missing = pd.MultiIndex.from_tuples([("asset", None), ("factor", "EUR")])
    with pytest.raises(ValueError, match="labels"):
        annual_arithmetic_from_log(mean.set_axis(missing), variance.set_axis(missing))


def test_finite_inputs_cannot_overflow_silently():
    """A finite input scale still has to yield a usable finite approximation."""
    mean = pd.Series([np.finfo(float).max], index=["asset"])
    variance = pd.Series([0.], index=mean.index)
    with pytest.raises(ValueError, match="approximation must be finite"):
        annual_arithmetic_from_log(mean, variance, 12.)
