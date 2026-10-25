"""Explicit supplied-moment preparation shared by opportunity acceptance tests."""
import pandas as pd

from factormodels.gls_diagnostic.opportunity import FactorOpportunityModel
from factormodels.estimation._results import snapshot_from_moments


def opportunity_from_moments(betas, variances, covariance, alpha, date=pd.Timestamp('2000-01-01'),
                             correlation=None, corr_weight=1., var_weight=1.):
    """Prepare risk through FL's boundary before constructing the diagnostic model."""
    snapshot = snapshot_from_moments(
        betas, covariance, variances, date=date, residual_correlation=correlation,
        residual_type='empirical' if correlation is not None else 'orthogonal',
        residual_corr_weight=corr_weight, residual_var_weight=var_weight,
    )
    return FactorOpportunityModel.from_snapshot(snapshot, alpha)
