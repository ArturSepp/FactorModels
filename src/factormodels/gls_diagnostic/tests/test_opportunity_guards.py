"""Opportunity input, risk attribution and solver failure contracts."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factormodels.gls_diagnostic import opportunity as op
from factormodels.gls_diagnostic.tests.test_opportunity import model


@pytest.mark.parametrize('field', ['betas', 'residual_variances', 'factor_covariance', 'alpha'])
def test_model_requires_labelled_inputs(field):
    """Unlabelled moments cannot silently acquire another asset or factor order."""
    m = model()
    with pytest.raises(ValueError, match=field + ' must be a labelled'):
        replace(m, **{field: getattr(m, field).to_numpy()})


def test_model_rejects_missing_date():
    """Risk snapshots cannot be keyed by an unavailable date."""
    with pytest.raises(ValueError, match='date must be finite'):
        replace(model(), date=pd.NaT)


@pytest.mark.parametrize('premium', [.05, 0.])
def test_factor_access_matches_direct_covariance_solution(premium):
    """A constant-beta model has a closed-form spectrum and systematic Sharpe."""
    m = model()
    summary, factors, spectrum = op.factor_access_metrics(
        m, pd.Series([premium], index=m.betas.columns))
    g = 3 * .09 / .04
    assert spectrum.g.iloc[0] == pytest.approx(g)
    assert spectrum.access_fraction.iloc[0] == pytest.approx(g / (1 + g))
    assert summary['factor_ceiling2'] == pytest.approx(premium**2 / .09)
    assert summary['systematic_sharpe2'] == pytest.approx(3 * premium**2 / (.04 + 3 * .09))
    assert summary['access_loss'] == pytest.approx(
        premium**2 / .09 - 3 * premium**2 / (.04 + 3 * .09))
    assert factors.tangency_direction.iloc[0] == pytest.approx(premium / .09)
    assert factors.premium_sensitivity.iloc[0] == pytest.approx(2 * premium / .09)
    assert factors.conditional_variance.iloc[0] == pytest.approx(.09)
    assert factors.drop_one_gain.iloc[0] == pytest.approx(premium**2 / .09)
    assert factors.hurdle.iloc[0] == pytest.approx(0.)
    if premium:
        assert summary['FPIR'] == pytest.approx(g / (1 + g))
    else:
        assert np.isnan(summary['FPIR'])
    no_premia, empty_factors, same_spectrum = op.factor_access_metrics(m)
    assert no_premia == {} and empty_factors.empty
    pd.testing.assert_frame_equal(spectrum, same_spectrum)


@pytest.mark.parametrize('variance', [0., -.01, np.nan, np.inf])
def test_candidate_gain_rejects_nonpositive_or_nonfinite_residual_risk(variance):
    """A candidate with undefined precision cannot enter the addition calculation."""
    m = model()
    premia = pd.Series(.05, index=m.betas.columns)
    with pytest.raises(ValueError, match='positive residual variance'):
        op.candidate_access_gain(m, premia, pd.Series(.6, index=premia.index), variance)


def test_omitted_reference_reports_absolute_saved_book_risk():
    """The reporting default is a zero book, preserving actual funding and absolute risk."""
    m = model()
    weights = pd.Series([.2, .3, .7], index=m.alpha.index)
    rows, exposures, holdings = op.portfolio_opportunity_metrics(m, weights)
    expected_variance = .09 * weights.sum()**2 + .04 * (weights @ weights)
    assert rows['net_exposure'] == pytest.approx(1.2)
    assert rows['reference_net_exposure'] == 0.
    assert rows['total_te']**2 == pytest.approx(expected_variance)
    assert rows['total_te'] == pytest.approx(rows['portfolio_vol'])
    assert rows['alpha_increment'] == pytest.approx(rows['alpha_level'])
    np.testing.assert_array_equal(holdings.reference, np.zeros(3))
    np.testing.assert_array_equal(exposures.exposure, exposures.active_exposure)
