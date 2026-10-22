# Canonical numerical API

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Use the [FactorModels package](https://github.com/ArturSepp/FactorModels) directly. Software citation: [CITATION.cff](../CITATION.cff). The root `factormodels.__all__` export list is authoritative. Parameters and full mathematical contracts are in each callable’s docstring.

## Choose an entry point

Start with `synthetic_bundle`, `load_bundle`, `fit_bundle` and
`FactorModelSnapshot` to obtain and inspect a model. `assemble_factor_model` and
`build_factor_panel` construct factor histories, and `FactorCovarEstimator`
exposes current and rolling estimation. `build_risk_model` exports a risk view.

For forecast analysis, use `infer_factor_premia`, `CmaDiagnosticInputs` and
`compute_cma_diagnostics` with explicit compatible units. The remaining premia,
identification, uncertainty and opportunity functions expose advanced analyses
with their own conditional assumptions. This grouping guides discovery within
the same public API. It does not change exports or imply a separate stability
guarantee for a pre-release package.

## `factormodels.spec`

- `RiskFactors`
- `FactorFamily`
- `VolTargetPolicy`
- `Factor`
- `FactorModelSpec`
- `DEFAULT_VOL_WARMUP_PERIOD`
- `assemble_factor_model`
- `normalise_factor_labels`

## `factormodels.construction`

- `build_factor_panel`
- `build_factor_catalogue_panel`
- `select_factor_prices`
- `compute_factor_nav`
- `compute_risk_factor_portfolio`
- `compute_live_strategic_weights`
- `compute_volatility_targeted_portfolio`
- `DEFAULT_RETURNS_FREQ`
- `INHERIT`

## `factormodels.constraints`

- `compute_beta_loading_signs`

The [construction guide](construction.md) states the completed definition/panel/constraint outputs and their downstream use. Catalogue and alias mappings are supplied by the caller; the library does not select a production model.

## `factormodels.premia`

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

## `factormodels.cma`

- `annual_arithmetic_from_log`
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

`annual_arithmetic_from_log(mean, variance, annualization_factor=1.0)` returns the
labelled second-order mean
approximation A × (mean + variance / 2). Inputs are finite, identically labelled
log-moment Series at the same cadence; variance is nonnegative. Use A=1 for moments
already annualised. It does not transform covariance or convert a regression
intercept in isolation. See the [complete mean bridge](gls_factor_premia.md).

## `factormodels.estimation`

- `FactorCovarEstimator`

## `factormodels.model`

- `DatedFactorCovariance`
- `FactorModelSnapshot`
- `RollingFactorModel`

`FactorModelSnapshot.loadings` and `.factor_covariance` return independent labelled
copies of the fitted tables. The existing `.asset_covariance` and
`.residual_covariance` retain the selected residual policy. These accessors perform
no estimation.

## `factormodels.examples`

- `synthetic_bundle`

Also exported from the package root. It returns fresh deterministic
synthetic `FactorBundle` inputs without fitting, downloading or saving anything.
Use `fit_bundle(synthetic_bundle())` from any directory after installing the
package. The [reference workflow](reference_workflows.md) reuses the same inputs.

## `factormodels.bundle`

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

## `factormodels.risk`

- `build_risk_model`

## Completed model contract

The [completed-output guide](completed_model.md) describes actual fields, mutable
pandas contents, supported inspection/export and the exact point at which fitting
is finished. [Reference workflows](reference_workflows.md) run without providers.

`FactorCovarEstimator.fit_current` requires `cutoff` and returns a `FactorModelSnapshot`; `fit_rolling` returns `RollingFactorModel`. Fitted FL diagnostics remain in `.data`. The selected total/residual covariance and native-factor metadata travel with the result. Export through `.to_risk_model()` or `build_risk_model` for existing QIS consumers. The risk view does not replace the complete fit.

Current fitting clips all historical inputs before estimation and does not mutate the caller’s FL configuration. `DatedFactorCovariance` declares upstream observation/availability dates; it is rejected when unavailable at the cutoff. Supplied dates are provenance declarations, not independently verified publication evidence.

GLS/CMA results condition on their supplied loadings/error model. Their intervals do not automatically include estimated-loading uncertainty or future-return risk. Correlated residual covariance is preserved in total risk; a diagonal-reference decomposition must be explicitly requested and labelled.

## `factormodels.portfolio_context`

- `prepare_portfolio_context`

The returned dictionary completes the risk/table handoff for a downstream portfolio
result. It contains no optimiser object and performs no portfolio solve.

## `factormodels.signals`

- `extract_rolling_clusters`
- `prepare_cluster_scores`

Completed mappings and pooled score panels can be consumed independently of the
fitted model. Scoring preserves the supplied point-in-time cluster schedule.

## `factormodels.analytics`

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
Statistical result tables remain FM outputs. Constrained portfolio solving is a
separate OP call. GLS diagnostics retain the stated diagonal-residual reference
geometry even when the selected full covariance includes residual correlation.

## Local paths

`get_paths` reads YAML settings. `get_resource_path` resolves input storage;
`get_output_path` creates and resolves output storage. See the
[local path configuration](reference_workflows.md).
