# Canonical numerical API

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Use the [TradableFactors package](https://github.com/ArturSepp/tradablefactors) directly. Software citation: [CITATION.cff](../CITATION.cff). The root `tradablefactors.__all__` export list is authoritative. Parameters and full mathematical contracts are in each callable’s docstring.

## `tradablefactors.premia`

- `GlsGeometry`
- `GlsPremiaResult`
- `GlsResolutionResult`
- `GlsUncertaintyScenario`
- `GlsLoadingStabilityResult`
- `ResidualOpportunity`
- `compute_gls_geometry`
- `compute_gls_resolution`
- `compute_gls_uncertainty`
- `compute_gls_reference_uncertainty`
- `compute_gls_adviser_disagreement`
- `compute_gls_provider_contrast`
- `compute_gls_loading_stability`
- `compute_gls_loading_stability_contrast`
- `summarize_gls_loading_stability`
- `compute_residual_opportunity`
- `infer_factor_premia`
- `factor_mimicking_portfolios`
- `gls_projector`
- `gls_split`
- `CompletionResult`
- `CompletionStatus`
- `achievable_sharpe2`
- `ceiling_premium_correlation_split`
- `completion_sharpe2`
- `factor_ceiling`
- `fixed_exposure_fpir`
- `fpir`
- `holdings_decomposition`
- `hurdle_premia`
- `ir_bound`
- `overlay_value`
- `partial_overlay_value`
- `tangency_access_split`

## `tradablefactors.cma`

- `AdmissionPolicy`
- `CmaDiagnosticInputs`
- `CmaDiagnosticResult`
- `CmaDiagnosticSpec`
- `compute_cma_diagnostics`
- `without_admitted_alpha`
- `CmaIdentificationResult`
- `compute_cma_identification`
- `CmaConditionalGlsResult`
- `compute_conditional_gls_intervals`

## `tradablefactors.estimation`

- `FactorCovarEstimator`

## `tradablefactors.model`

- `DatedFactorCovariance`
- `FactorModelSnapshot`
- `RollingFactorModel`

## `tradablefactors.risk`

- `build_risk_model`

## Completed model contract

`FactorCovarEstimator.fit_current` requires `cutoff` and returns a `FactorModelSnapshot`; `fit_rolling` returns `RollingFactorModel`. Fitted FL diagnostics remain in `.data`. The selected total/residual covariance and native-factor metadata travel with the result. Export through `.to_risk_model()` or `build_risk_model` for existing QIS consumers. The risk view does not replace the complete fit.

Current fitting clips all historical inputs before estimation and does not mutate the caller’s FL configuration. `DatedFactorCovariance` declares upstream observation/availability dates; it is rejected when unavailable at the cutoff. Supplied dates are provenance declarations, not independently verified publication evidence.

GLS/CMA results condition on their supplied loadings/error model. Their intervals do not automatically include estimated-loading uncertainty or future-return risk. Correlated residual covariance is preserved in total risk; a diagonal-reference decomposition must be explicitly requested and labelled.
