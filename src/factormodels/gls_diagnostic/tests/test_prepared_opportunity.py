"""Complete labelled optimisation handoff without importing a consumer package."""

from factormodels.gls_diagnostic.tests._fixtures import opportunity_from_moments

import numpy as np
import pandas as pd
import pytest
from factorlasso import ResidualCorrelationData

from factormodels.gls_diagnostic.tests.test_alpha_uncertainty import model


def test_prepared_arrays_match_independent_risk_and_projection():
    """Upstream GLS yields complete risk and orthogonal objective arrays."""
    fitted = model()
    inputs = fitted.optimisation_inputs()
    design = np.column_stack([np.ones(len(fitted.alpha)), fitted.betas])
    projector = np.eye(len(design)) - design @ np.linalg.pinv(design)
    expected = fitted.betas @ fitted.factor_covariance @ fitted.betas.T
    expected += np.diag(fitted.residual_variances)
    pd.testing.assert_frame_equal(inputs['covariance'], expected)
    np.testing.assert_allclose(inputs['orthogonal_alpha'], projector @ fitted.alpha, atol=1e-15)
    assert inputs['date'] == fitted.date
    inputs['alpha'].iloc[0] = 100.
    assert fitted.alpha.iloc[0] != 100.


def test_uncertainty_projection_is_prepared_upstream_and_keeps_native_units():
    """A supplied estimation covariance is projected once, without return scaling."""
    fitted = model()
    ids = fitted.alpha.index
    covariance = pd.DataFrame(np.diag([.001, .002, .003, .004]), index=ids, columns=ids)
    design = np.column_stack([np.ones(len(ids)), fitted.betas])
    projector = np.eye(len(ids)) - design @ np.linalg.pinv(design)
    pd.testing.assert_frame_equal(fitted.prepare_alpha_covariance(covariance), covariance)
    np.testing.assert_allclose(fitted.prepare_alpha_covariance(covariance, orthogonal=True),
                               projector @ covariance @ projector.T, atol=1e-17)
    with pytest.raises(ValueError, match='exact asset order'):
        fitted.prepare_alpha_covariance(covariance.iloc[::-1])
    with pytest.raises(ValueError):
        fitted.prepare_alpha_covariance(-covariance)


def test_correlated_residual_policy_survives_the_plain_array_handoff():
    """The exported covariance includes selected off-diagonal residual dependence."""
    ids = pd.Index(['a', 'b'])
    date = pd.Timestamp('2026-06-30')
    correlation = ResidualCorrelationData(
        pd.DataFrame([[1., .8], [.8, 1.]], index=ids, columns=ids),
        pd.DataFrame([[1., 1.]], index=[date], columns=ids), pd.DataFrame(index=ids),
        'QE', 40., date, date)
    fitted = opportunity_from_moments(pd.DataFrame(1., index=ids, columns=['market']),
        pd.Series(.04, index=ids), pd.DataFrame(.09, index=['market'], columns=['market']),
        pd.Series([0., .04], index=ids), date, correlation, .5)
    inputs = fitted.optimisation_inputs()
    residual = .04 * np.array([[1., .4], [.4, 1.]])
    np.testing.assert_allclose(inputs['residual_covariance'], residual)
    np.testing.assert_allclose(inputs['covariance'], residual + .09)
    zero = opportunity_from_moments(fitted.betas, fitted.residual_variances, fitted.factor_covariance,
        fitted.alpha, date, correlation, 0.).optimisation_inputs()
    np.testing.assert_allclose(zero['residual_covariance'], np.eye(2)*.04)
