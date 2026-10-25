"""Independent numerical contracts for statistical-alpha opportunity analytics."""
from factormodels.gls_diagnostic.tests._fixtures import opportunity_from_moments

import numpy as np
import pandas as pd
import pytest
from factormodels.gls_diagnostic.opportunity import alpha_dispersion, candidate_access_gain
from factormodels.gls_diagnostic import achievable_sharpe2


def model():
    """A small model with an exact self-financing, factor-neutral direction."""
    names = pd.Index(['a', 'b', 'c'])
    return opportunity_from_moments(
        pd.DataFrame([1., 1., 1.], index=names, columns=['market']),
        pd.Series(0.04, index=names),
        pd.DataFrame([[0.09]], index=['market'], columns=['market']),
        pd.Series([-0.02, 0., 0.02], index=names),
    )


def test_joint_dispersion_and_affine_invariance():
    """Compare K with an independent whitened SVD and remove affine alpha."""
    m = model()
    result, detail = alpha_dispersion(m)
    c = np.column_stack([np.ones(3), m.betas]) / np.sqrt(m.residual_variances.to_numpy())[:, None]
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    rank = np.sum(s > s[0] * 1e-12)
    aw = m.alpha.to_numpy() / np.sqrt(m.residual_variances)
    residual = aw - u[:, :rank] @ (u[:, :rank].T @ aw)
    assert result['K'] == pytest.approx(residual @ residual, abs=1e-12)
    assert result['K'] == pytest.approx(0.02)
    assert result['K'] <= min(result['J'], result['S_h2']) + 1e-12
    shifted = opportunity_from_moments(m.betas, m.residual_variances, m.factor_covariance, m.alpha + 0.3)
    assert alpha_dispersion(shifted)[0]['K'] == pytest.approx(result['K'])
    assert np.max(np.abs(c.T @ (detail.h_joint / np.sqrt(m.residual_variances)))) < 1e-12






def test_candidate_gain_matches_direct_refit_free_difference():
    """Sherman-Morrison gain agrees with the canonical expanded-universe value."""
    m = model()
    premia = pd.Series([0.05], index=m.betas.columns)
    candidate = pd.Series([0.6], index=m.betas.columns)
    gain = candidate_access_gain(m, premia, candidate, 0.02)
    expanded_b = pd.concat([m.betas, pd.DataFrame([candidate], index=['new'])])
    expanded_d = pd.concat([m.residual_variances, pd.Series([0.02], index=['new'])])
    direct = achievable_sharpe2(premia, m.factor_covariance, expanded_b, expanded_d)
    direct -= achievable_sharpe2(premia, m.factor_covariance, m.betas, m.residual_variances)
    assert gain == pytest.approx(direct, abs=1e-12)


def test_labels_variances_and_factor_covariance_are_not_repaired():
    """Input failures cannot silently change the investment model."""
    m = model()
    with pytest.raises(ValueError):
        opportunity_from_moments(m.betas, m.residual_variances.iloc[::-1], m.factor_covariance, m.alpha)
    with pytest.raises(ValueError):
        opportunity_from_moments(m.betas, m.residual_variances * 0, m.factor_covariance, m.alpha)
    with pytest.raises(ValueError):
        opportunity_from_moments(m.betas, m.residual_variances, m.factor_covariance * 0, m.alpha)










def test_normalized_capacity_and_common_level_dispersion():
    """Replicating comparable independent modeled assets scales totals, not RMS."""
    m = model()
    base, _ = alpha_dispersion(m)
    b = pd.concat([m.betas, m.betas], ignore_index=True)
    doubled = opportunity_from_moments(b, pd.concat([m.residual_variances] * 2, ignore_index=True),
                               m.factor_covariance, pd.concat([m.alpha] * 2, ignore_index=True))
    other, _ = alpha_dispersion(doubled)
    for name in ('A', 'J', 'K'):
        assert other[name] == pytest.approx(2 * base[name])
        assert other[name + '_rms'] == pytest.approx(base[name + '_rms'])
    constant = opportunity_from_moments(m.betas, m.residual_variances, m.factor_covariance,
                                pd.Series(.02, index=m.betas.index))
    result, _ = alpha_dispersion(constant)
    assert result['A'] > 0
    assert result['J'] == pytest.approx(0., abs=1e-14)
    assert result['J_weighted_alpha_std'] == pytest.approx(0., abs=1e-14)


def test_all_negative_alpha_keeps_signed_reference_and_zero_positive_capacity():
    """Positive-part capacity never converts a negative alpha universe into gains."""
    m = model()
    m = opportunity_from_moments(m.betas, m.residual_variances, m.factor_covariance, m.alpha-.1)
    rows, detail = alpha_dispersion(m)
    assert rows['A_plus'] == 0.
    assert rows['residual_reference_signed_sharpe'] == pytest.approx(-.4)
    assert detail.residual_reference_weight['c'] == 1.
