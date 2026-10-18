"""Canonical factor-model, GLS and CMA numerical API."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("tradablefactors")
except _PackageNotFoundError:
    __version__ = "0+unknown"

from tradablefactors.premia import (
    GlsGeometry,
    GlsPremiaResult,
    GlsResolutionResult,
    GlsUncertaintyScenario,
    GlsLoadingStabilityResult,
    ResidualOpportunity,
    compute_gls_geometry,
    compute_gls_resolution,
    compute_gls_uncertainty,
    compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement,
    compute_gls_provider_contrast,
    compute_gls_loading_stability,
    compute_gls_loading_stability_contrast,
    summarize_gls_loading_stability,
    compute_residual_opportunity,
    infer_factor_premia,
    factor_mimicking_portfolios,
    gls_projector,
    gls_split,
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

from tradablefactors.cma import (
    AdmissionPolicy,
    CmaDiagnosticInputs,
    CmaDiagnosticResult,
    CmaDiagnosticSpec,
    compute_cma_diagnostics,
    without_admitted_alpha,
    CmaIdentificationResult,
    compute_cma_identification,
    CmaConditionalGlsResult,
    compute_conditional_gls_intervals,
)

from tradablefactors.estimation import (
    FactorCovarEstimator,
)

from tradablefactors.model import (
    DatedFactorCovariance,
    FactorModelSnapshot,
    RollingFactorModel,
)

from tradablefactors.risk import (
    build_risk_model,
)

from tradablefactors.spec import (
    DEFAULT_VOL_WARMUP_PERIOD,
    RiskFactors,
    FactorFamily,
    VolTargetPolicy,
    Factor,
    FactorModelSpec,
    assemble_factor_model,
    normalise_factor_labels,
)

from tradablefactors.construction import (
    DEFAULT_RETURNS_FREQ,
    INHERIT,
    build_factor_panel,
    build_factor_catalogue_panel,
    select_factor_prices,
    compute_factor_nav,
    compute_risk_factor_portfolio,
    compute_live_strategic_weights,
    compute_volatility_targeted_portfolio,
)

from tradablefactors.constraints import (
    compute_beta_loading_signs,
)

from tradablefactors.analytics import (
    FactorOpportunityModel,
    alpha_dispersion,
    factor_access_metrics,
    candidate_access_gain,
    portfolio_opportunity_metrics,
    alpha_uncertainty_metrics,
    removal_uncertainty,
    fixed_portfolio_alpha_interval,
    subset_dispersion,
    dispersion_concentration,
    removal_diagnostics,
    bundle_removal_diagnostics,
    compare_universe_dispersion,
    sample_universe_dispersion,
)
from tradablefactors.portfolio_context import prepare_portfolio_context
from tradablefactors.signals import extract_rolling_clusters, prepare_cluster_scores
from tradablefactors.bundle import (
    BUNDLE_SCHEMA_VERSION, BundleSettings, FactorBundle, load_bundle, fit_bundle, save_bundle,
)

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleSettings",
    "FactorBundle",
    "load_bundle",
    "fit_bundle",
    "save_bundle",
    "FactorOpportunityModel",
    "alpha_dispersion",
    "factor_access_metrics",
    "candidate_access_gain",
    "portfolio_opportunity_metrics",
    "alpha_uncertainty_metrics",
    "removal_uncertainty",
    "fixed_portfolio_alpha_interval",
    "subset_dispersion",
    "dispersion_concentration",
    "removal_diagnostics",
    "bundle_removal_diagnostics",
    "compare_universe_dispersion",
    "sample_universe_dispersion",
    "prepare_portfolio_context",
    "extract_rolling_clusters",
    "prepare_cluster_scores",

    "GlsGeometry",
    "GlsPremiaResult",
    "GlsResolutionResult",
    "GlsUncertaintyScenario",
    "GlsLoadingStabilityResult",
    "ResidualOpportunity",
    "compute_gls_geometry",
    "compute_gls_resolution",
    "compute_gls_uncertainty",
    "compute_gls_reference_uncertainty",
    "compute_gls_adviser_disagreement",
    "compute_gls_provider_contrast",
    "compute_gls_loading_stability",
    "compute_gls_loading_stability_contrast",
    "summarize_gls_loading_stability",
    "compute_residual_opportunity",
    "infer_factor_premia",
    "factor_mimicking_portfolios",
    "gls_projector",
    "gls_split",
    "CompletionResult",
    "CompletionStatus",
    "achievable_sharpe2",
    "ceiling_premium_correlation_split",
    "completion_sharpe2",
    "factor_ceiling",
    "fixed_exposure_fpir",
    "fpir",
    "holdings_decomposition",
    "hurdle_premia",
    "ir_bound",
    "overlay_value",
    "partial_overlay_value",
    "tangency_access_split",
    "AdmissionPolicy",
    "CmaDiagnosticInputs",
    "CmaDiagnosticResult",
    "CmaDiagnosticSpec",
    "compute_cma_diagnostics",
    "without_admitted_alpha",
    "CmaIdentificationResult",
    "compute_cma_identification",
    "CmaConditionalGlsResult",
    "compute_conditional_gls_intervals",
    "FactorCovarEstimator",
    "DatedFactorCovariance",
    "FactorModelSnapshot",
    "RollingFactorModel",
    "build_risk_model",
    "DEFAULT_VOL_WARMUP_PERIOD",
    "RiskFactors",
    "FactorFamily",
    "VolTargetPolicy",
    "Factor",
    "FactorModelSpec",
    "assemble_factor_model",
    "normalise_factor_labels",
    "DEFAULT_RETURNS_FREQ",
    "INHERIT",
    "build_factor_panel",
    "build_factor_catalogue_panel",
    "select_factor_prices",
    "compute_factor_nav",
    "compute_risk_factor_portfolio",
    "compute_live_strategic_weights",
    "compute_volatility_targeted_portfolio",
    "compute_beta_loading_signs",
]
