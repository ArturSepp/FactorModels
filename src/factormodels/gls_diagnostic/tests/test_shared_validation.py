"""Shared validation tightens only declared labels/dtypes and preserves full whitening."""
import numpy as np
import pandas as pd
import pytest

from factormodels.gls_diagnostic import sharpe_accounting as sa
from factormodels.gls_diagnostic.geometry import compute_gls_geometry
from factormodels.gls_diagnostic.inference import infer_factor_premia
from factormodels.gls_diagnostic._validation import _labelled, _whiten_loadings


@pytest.fixture
def moments():
    """Supply aligned labelled moments for three assets and two factors."""
    assets = pd.Index(["a", "b", "c"])
    factors = pd.Index(["equity", "rates"])
    b = pd.DataFrame([[1., .2], [.3, 1.], [-.4, .6]], index=assets, columns=factors)
    d = pd.Series([.01, .03, .02], index=assets)
    f = pd.DataFrame([[.04, .003], [.003, .01]], index=factors, columns=factors)
    lam = pd.Series([.05, .02], index=factors)
    return lam, f, b, d


@pytest.mark.parametrize("missing", [None, np.nan])
def test_matching_missing_factor_labels_are_rejected(moments, missing):
    """Matching missing labels are invalid even when both axes agree."""
    lam, f, _, _ = moments
    labels = pd.Index(["equity", missing])
    lam.index = labels
    f.index = f.columns = labels
    with pytest.raises(ValueError, match="label/order"):
        sa.factor_ceiling(lam, f)


@pytest.mark.parametrize("axis", ["assets", "factors", "residuals"])
def test_missing_loading_and_covariance_labels_are_rejected(moments, axis):
    """Reject missing labels on either loading axis or the residual covariance."""
    lam, f, b, d = moments
    if axis == "assets":
        b.index = d.index = pd.Index(["a", "b", np.nan])
    elif axis == "factors":
        b.columns = pd.Index(["equity", None])
    else:
        b = b.to_numpy()
        d.index = pd.Index(["a", "b", np.nan])
    with pytest.raises(ValueError, match="label/order"):
        sa.achievable_sharpe2(lam, f, b, d)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_labelled_object_numeric_inputs_are_rejected(moments, position):
    """Require numerical pandas dtypes at every labelled accounting input."""
    inputs = list(moments)
    inputs[position] = inputs[position].astype(object)
    with pytest.raises(ValueError, match="finite real values"):
        sa.achievable_sharpe2(*inputs)


def test_geometry_and_forecast_vectors_share_dtype_validation(moments):
    """Geometry and inference reject string or object arrays with pandas labels."""
    _, _, b, d = moments
    with pytest.raises(ValueError, match="finite real values"):
        compute_gls_geometry(b.astype(str), d)
    with pytest.raises(ValueError, match="finite real values"):
        compute_gls_geometry(b, pd.DataFrame(np.diag(d), index=b.index, columns=b.index).astype(object))
    with pytest.raises(ValueError, match="finite real values"):
        infer_factor_premia(b, d, pd.Series([".02", ".03", ".01"], index=b.index), cash_rate=0.)


def test_unlabelled_conversion_and_descriptive_tables_remain_supported(moments):
    """Keep explicit array coercion and nonnumerical metadata contracts intact."""
    lam, f, b, d = moments
    expected = sa.achievable_sharpe2(lam, f, b, d)
    actual = sa.achievable_sharpe2(lam.astype(str).tolist(), f.to_numpy().astype(str),
                                  b.to_numpy(), d.to_numpy())
    assert actual == expected
    descriptions = pd.DataFrame({"sector": ["equity", "bond", "credit"]}, index=b.index)
    assert _labelled(descriptions, pd.DataFrame, b.index, "metadata", finite=False) is descriptions


def test_established_alignment_error_messages_survive_consolidation(moments):
    """Retain each public operation's existing alignment error text."""
    lam, f, b, d = moments
    with pytest.raises(ValueError) as exc:
        sa.achievable_sharpe2(lam.iloc[::-1], f, b, d)
    assert str(exc.value) == "label/order mismatch or duplicate labels"
    with pytest.raises(ValueError) as exc:
        compute_gls_geometry(b, d.iloc[::-1])
    assert str(exc.value) == "label/order mismatch or duplicate/missing labels"
    with pytest.raises(ValueError) as exc:
        sa.overlay_value(lam, f, b, d, pd.Series([.01, .02, .03], index=b.index[::-1]))
    assert str(exc.value) == "label/order mismatch or duplicate labels"


def test_accounting_does_not_call_svd_or_truncate_whitened_loadings(monkeypatch, moments):
    """Accounting preserves tiny directions and agrees with an asset-space solve."""
    lam, f, b, d = moments
    mean = b.to_numpy() @ lam.to_numpy()
    expected = mean @ np.linalg.solve(b.to_numpy() @ f.to_numpy() @ b.to_numpy().T + np.diag(d), mean)
    def forbidden(*args, **kwargs):
        """Fail if accounting delegates whitening to rank-revealing geometry."""
        raise AssertionError("accounting validation must not call the geometry SVD")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    assert sa.achievable_sharpe2(lam, f, b, d) == pytest.approx(expected, abs=1e-12, rel=1e-12)
    tiny = np.zeros((4, 2))
    tiny[0, 0], tiny[1, 1] = 1., 1e-18
    np.testing.assert_array_equal(_whiten_loadings(tiny, np.ones(4))[-1], tiny)


def test_geometry_rank_cutoff_does_not_change_accounting_preparation():
    """A geometric rank cutoff must not remove accounting loading directions."""
    b = np.zeros((4, 2))
    b[0, 0], b[1, 1] = 1., 1e-18
    assert compute_gls_geometry(b, np.ones(4)).rank == 1
    white = sa._model(np.ones(2), np.eye(2), b, np.ones(4))[-1]
    np.testing.assert_array_equal(white, b)
    assert white[1, 1] > 0
