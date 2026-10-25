"""A completed public diagnostic object produces a real, noninteractive report."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factormodels.gls_diagnostic import CmaDiagnosticInputs, compute_cma_diagnostics
from factormodels.reports.cma_diagnostics import render_cma_diagnostics


def test_completed_cma_result_renders_without_changing_its_tables(tmp_path):
    """Rendering consumes completed results and preserves their numerical contents."""
    assets, factors = list('abcd'), ['Equity', 'Rates']
    betas = pd.DataFrame([[1., 0.], [0., 1.], [.5, .6], [1., -.2]],
                         index=assets, columns=factors)
    residual = pd.Series([.01, .02, .03, .04], index=assets)
    covariance = pd.DataFrame([[.04, .005], [.005, .01]], index=factors, columns=factors)
    premia = pd.Series([.05, .02], index=factors)
    forecast = betas @ premia + pd.Series([.01, .015, -.01, .003], index=assets)
    result = compute_cma_diagnostics(CmaDiagnosticInputs(
        betas, residual, forecast, covariance, premia,
        provenance={'reference_ccy': 'USD', 'as_of': '2026-09-30'}))
    saved = deepcopy(result)
    figures_before = set(plt.get_fignums())
    output = render_cma_diagnostics(result, tmp_path / 'cma.pdf', pages=2)
    contents = output.read_bytes()
    assert contents.startswith(b'%PDF-') and len(contents) > 1000
    assert set(plt.get_fignums()) == figures_before
    pd.testing.assert_frame_equal(result.assets, saved.assets)
    pd.testing.assert_frame_equal(result.factors, saved.factors)
    np.testing.assert_equal(result.summary, saved.summary)
