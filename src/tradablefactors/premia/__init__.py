"""GLS risk-premia inference conditional on an estimated factor model.

Provider CMAs, cash, alpha admission and forecast uncertainty belong to this
factor-analytics layer. Historical loading estimation remains in FactorLasso.
"""

from tradablefactors.premia._inference import (
    GlsGeometry,
    GlsPremiaResult,
    compute_gls_geometry,
    factor_mimicking_portfolios,
    gls_projector,
    gls_split,
    infer_factor_premia,
)
from tradablefactors.premia._opportunity import (
    ResidualOpportunity,
    compute_residual_opportunity,
)
from tradablefactors.premia._resolution import (
    GlsResolutionResult,
    compute_gls_resolution,
)
from tradablefactors.premia._scenarios import (
    GlsUncertaintyScenario,
    compute_gls_uncertainty,
    compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement,
    compute_gls_provider_contrast,
)
from tradablefactors.premia._stability import (
    GlsLoadingStabilityResult,
    summarize_gls_loading_stability,
    compute_gls_loading_stability,
    compute_gls_loading_stability_contrast,
)

from tradablefactors.premia.sharpe_accounting import (
    CompletionResult,
    CompletionStatus,
    achievable_sharpe2,
    ceiling_premium_correlation_split,
    completion_sharpe2,
    factor_ceiling,
    fixed_exposure_fpir,
    fpir,
    holdings_decomposition,
    hurdle_premia,
    ir_bound,
    overlay_value,
    partial_overlay_value,
    tangency_access_split,
)

__all__ = [
    'GlsGeometry', 'GlsPremiaResult', 'GlsResolutionResult', 'GlsUncertaintyScenario',
    'GlsLoadingStabilityResult', 'ResidualOpportunity', 'compute_gls_geometry',
    'compute_gls_resolution', 'compute_gls_uncertainty', 'compute_gls_reference_uncertainty',
    'compute_gls_adviser_disagreement', 'compute_gls_provider_contrast',
    'compute_gls_loading_stability', 'compute_gls_loading_stability_contrast',
    'summarize_gls_loading_stability', 'compute_residual_opportunity', 'infer_factor_premia',
    'factor_mimicking_portfolios', 'gls_projector', 'gls_split', 'CompletionResult',
    'CompletionStatus', 'achievable_sharpe2', 'ceiling_premium_correlation_split',
    'completion_sharpe2', 'factor_ceiling', 'fixed_exposure_fpir', 'fpir',
    'holdings_decomposition', 'hurdle_premia', 'ir_bound', 'overlay_value',
    'partial_overlay_value', 'tangency_access_split',
]
