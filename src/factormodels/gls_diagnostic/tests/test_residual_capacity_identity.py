"""Independent capacity/noise arithmetic agrees for an identical funding-plus-factor design."""
import numpy as np
import pandas as pd
import pytest

from factormodels.gls_diagnostic.residual_opportunity import compute_residual_opportunity
from factormodels.gls_diagnostic.opportunity import alpha_dispersion
from factormodels.gls_diagnostic.alpha_uncertainty import alpha_uncertainty_metrics
from factormodels.gls_diagnostic.universe_diagnostics import _subset_summary
from factormodels.gls_diagnostic.tests._fixtures import opportunity_from_moments


@pytest.mark.parametrize("duplicate_factor", [False, True])
def test_capacity_and_noise_floor_match_with_explicit_funding(duplicate_factor):
    """Independent projections agree with both full and duplicated factor designs."""
    rng = np.random.default_rng(20261025)
    ids, factors = pd.Index([f"a{i}" for i in range(12)]), pd.Index(["equity", "rates"])
    b = pd.DataFrame(rng.normal(size=(12, 2)), index=ids, columns=factors)
    if duplicate_factor:
        b["rates"] = b["equity"]
    d = pd.Series(rng.uniform(.01, .08, 12), index=ids)
    a = pd.Series(rng.normal(0, .02, 12), index=ids)
    v0 = rng.normal(0, .004, (12, 12))
    v = pd.DataFrame(v0 @ v0.T, index=ids, columns=ids)
    funding = pd.DataFrame({"funding": np.ones(12)}, index=ids)
    model = opportunity_from_moments(b, d, pd.DataFrame(np.diag([.04, .01]), index=factors, columns=factors), a)
    primitive = compute_residual_opportunity(b, d, a, controls=funding, alpha_covariance=v)
    labelled, _ = alpha_dispersion(model)
    subset = _subset_summary(model, ids, v)
    uncertainty = alpha_uncertainty_metrics(model, v, draws=8, seed=17)
    # Same span, but independent operation order: whitened norm/trace versus
    # labelled sums and quadratic covariance propagation. Do not merge them.
    np.testing.assert_allclose(primitive.summary["capacity"], labelled["K"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(primitive.summary["noise_floor"], subset["K_noise"], atol=1e-10, rtol=1e-10)
    summary = uncertainty["summary"]
    row = summary.loc["K"] if "K" in summary.index else summary.loc[summary.metric == "K"].iloc[0]
    np.testing.assert_allclose(primitive.summary["noise_floor"], row["noise"], atol=1e-10, rtol=1e-10)
    assert primitive.summary["neutral_dimension"] == len(ids) - (2 if duplicate_factor else 3)
