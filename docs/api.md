# Canonical numerical API

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Use the [TradableFactors package](https://github.com/ArturSepp/tradablefactors) directly. Software citation: [CITATION.cff](../CITATION.cff). The root `tradablefactors.__all__` export list is authoritative. Parameters and full mathematical contracts are in each callable’s docstring.

## `tradablefactors.spec`

- `RiskFactors`
- `FactorFamily`
- `VolTargetPolicy`
- `Factor`
- `FactorModelSpec`
- `DEFAULT_VOL_WARMUP_PERIOD`
- `assemble_factor_model`
- `normalise_factor_labels`

## `tradablefactors.construction`

- `build_factor_panel`
- `build_factor_catalogue_panel`
- `select_factor_prices`
- `compute_factor_nav`
- `compute_risk_factor_portfolio`
- `compute_live_strategic_weights`
- `compute_volatility_targeted_portfolio`
- `DEFAULT_RETURNS_FREQ`
- `INHERIT`

## `tradablefactors.constraints`

- `compute_beta_loading_signs`

The [construction guide](construction.md) states the completed definition/panel/constraint outputs and their downstream use. Catalogue and alias mappings are supplied by the caller; the library does not select a production model.

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

## `tradablefactors.bundle`

- `BUNDLE_SCHEMA_VERSION`
- `BundleSettings`
- `FactorBundle`
- `load_bundle`
- `fit_bundle`
- `save_bundle`

The [six-file contract](rolling_factor_covar_from_csv.md) defines schema version 1,
units and configuration. Loading produces validated inputs only. Fitting returns
a snapshot or rolling result; saving inputs creates a new directory. There is no
implicit acquisition, fitted-result checkpoint or historical-format adapter.

## `tradablefactors.risk`

- `build_risk_model`

## Completed model contract

The [completed-output guide](completed_model.md) describes actual fields, mutable
pandas contents, supported inspection/export and the exact point at which fitting
is finished. [Reference workflows](reference_workflows.md) run without providers.

`FactorCovarEstimator.fit_current` requires `cutoff` and returns a `FactorModelSnapshot`; `fit_rolling` returns `RollingFactorModel`. Fitted FL diagnostics remain in `.data`. The selected total/residual covariance and native-factor metadata travel with the result. Export through `.to_risk_model()` or `build_risk_model` for existing QIS consumers. The risk view does not replace the complete fit.

Current fitting clips all historical inputs before estimation and does not mutate the caller’s FL configuration. `DatedFactorCovariance` declares upstream observation/availability dates; it is rejected when unavailable at the cutoff. Supplied dates are provenance declarations, not independently verified publication evidence.

GLS/CMA results condition on their supplied loadings/error model. Their intervals do not automatically include estimated-loading uncertainty or future-return risk. Correlated residual covariance is preserved in total risk; a diagonal-reference decomposition must be explicitly requested and labelled.

## `tradablefactors.portfolio_context`

- `prepare_portfolio_context`

The returned dictionary completes the risk/table handoff for a downstream portfolio
result. It contains no optimiser object and performs no portfolio solve.

## `tradablefactors.signals`

- `extract_rolling_clusters`
- `prepare_cluster_scores`

Completed mappings and pooled score panels can be consumed independently of the
fitted model. Scoring preserves the supplied point-in-time cluster schedule.

## `tradablefactors.analytics`

- `FactorOpportunityModel`
- `alpha_dispersion`
- `factor_access_metrics`
- `candidate_access_gain`
- `portfolio_opportunity_metrics`
- `alpha_uncertainty_metrics`
- `removal_uncertainty`
- `fixed_portfolio_alpha_interval`
- `subset_dispersion`
- `dispersion_concentration`
- `removal_diagnostics`
- `bundle_removal_diagnostics`
- `compare_universe_dispersion`
- `sample_universe_dispersion`

`FactorOpportunityModel.optimisation_inputs()` produces copied labelled arrays;
`prepare_alpha_covariance` prepares the chosen objective's estimation covariance.
Statistical result tables remain TF outputs. Constrained portfolio solving is a
separate OP call. GLS diagnostics retain the stated diagonal-residual reference
geometry even when the selected full covariance includes residual correlation.
