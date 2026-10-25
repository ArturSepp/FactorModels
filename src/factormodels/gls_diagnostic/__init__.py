"""Completed GLS geometry, inference, uncertainty, opportunity and CMA exports.

Validation/whitening sits below geometry and Sharpe accounting. Historical alpha
uncertainty, conditional CMA inference, supplied forecast uncertainty and loading
sensitivity have distinct owners and conditioning assumptions.
"""
from factormodels.gls_diagnostic.geometry import (
    GlsGeometry,
    compute_gls_geometry,
    factor_mimicking_portfolios,
    gls_projector,
    gls_split,
)
from factormodels.gls_diagnostic.inference import GlsPremiaResult, infer_factor_premia
from factormodels.gls_diagnostic.residual_opportunity import ResidualOpportunity, compute_residual_opportunity
from factormodels.gls_diagnostic.resolution import GlsResolutionResult, compute_gls_resolution
from factormodels.gls_diagnostic.forecast_uncertainty import (
    GlsUncertaintyScenario,
    compute_gls_uncertainty,
    compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement,
    compute_gls_provider_contrast,
)
from factormodels.gls_diagnostic.loading_stability import (
    GlsLoadingStabilityResult,
    summarize_gls_loading_stability,
    compute_gls_loading_stability,
    compute_gls_loading_stability_contrast,
)
from factormodels.gls_diagnostic.sharpe_accounting import (
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
from factormodels.gls_diagnostic.cma_diagnostics import (
    AdmissionPolicy,
    CmaDiagnosticInputs,
    CmaDiagnosticResult,
    CmaDiagnosticSpec,
    compute_cma_diagnostics,
    without_admitted_alpha,
)
from factormodels.gls_diagnostic.cma_identification import CmaIdentificationResult, compute_cma_identification
from factormodels.gls_diagnostic.cma_moments import annual_arithmetic_from_log
from factormodels.gls_diagnostic.cma_inference import CmaConditionalGlsResult, compute_conditional_gls_intervals
from factormodels.gls_diagnostic.opportunity import (
    FactorOpportunityModel,
    alpha_dispersion,
    factor_access_metrics,
    candidate_access_gain,
    portfolio_opportunity_metrics,
)
from factormodels.gls_diagnostic.alpha_uncertainty import (
    alpha_uncertainty_metrics,
    removal_uncertainty,
    fixed_portfolio_alpha_interval,
)
from factormodels.gls_diagnostic.universe_diagnostics import (
    subset_dispersion,
    dispersion_concentration,
    removal_diagnostics,
    bundle_removal_diagnostics,
    compare_universe_dispersion,
    sample_universe_dispersion,
)

__all__ = [
    'GlsGeometry',
    'GlsPremiaResult',
    'compute_gls_geometry',
    'factor_mimicking_portfolios',
    'gls_projector',
    'gls_split',
    'infer_factor_premia',
    'ResidualOpportunity',
    'compute_residual_opportunity',
    'GlsResolutionResult',
    'compute_gls_resolution',
    'GlsUncertaintyScenario',
    'compute_gls_uncertainty',
    'compute_gls_reference_uncertainty',
    'compute_gls_adviser_disagreement',
    'compute_gls_provider_contrast',
    'GlsLoadingStabilityResult',
    'summarize_gls_loading_stability',
    'compute_gls_loading_stability',
    'compute_gls_loading_stability_contrast',
    'CompletionResult',
    'CompletionStatus',
    'achievable_sharpe2',
    'ceiling_premium_correlation_split',
    'completion_sharpe2',
    'factor_ceiling',
    'fixed_exposure_fpir',
    'fpir',
    'holdings_decomposition',
    'hurdle_premia',
    'ir_bound',
    'overlay_value',
    'partial_overlay_value',
    'tangency_access_split',
    'AdmissionPolicy',
    'CmaDiagnosticInputs',
    'CmaDiagnosticResult',
    'CmaDiagnosticSpec',
    'compute_cma_diagnostics',
    'without_admitted_alpha',
    'CmaIdentificationResult',
    'compute_cma_identification',
    'annual_arithmetic_from_log',
    'CmaConditionalGlsResult',
    'compute_conditional_gls_intervals',
    'FactorOpportunityModel',
    'alpha_dispersion',
    'factor_access_metrics',
    'candidate_access_gain',
    'portfolio_opportunity_metrics',
    'alpha_uncertainty_metrics',
    'removal_uncertainty',
    'fixed_portfolio_alpha_interval',
    'subset_dispersion',
    'dispersion_concentration',
    'removal_diagnostics',
    'bundle_removal_diagnostics',
    'compare_universe_dispersion',
    'sample_universe_dispersion',
]
