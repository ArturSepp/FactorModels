---
myst:
  html_meta:
    description: >-
      Canonical numerical API of factormodels: the public entry points for factor construction, estimation, residual policies, GLS premia, CMA diagnostics and supplied histories.
---

# Canonical numerical API

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Use the [FactorModels package](https://github.com/ArturSepp/FactorModels) directly. The root `factormodels.__all__` export list is authoritative. Parameters and full mathematical contracts are in each callable’s docstring.

## Choose an entry point

Start with `FactorCovarEstimator.fit_current` or `.fit_rolling` and
`FactorModelSnapshot` to estimate and inspect a model from DataFrames. `assemble_factor_model` and
`build_factor_panel` construct factor histories, and `FactorCovarEstimator`
exposes current and rolling estimation. `build_risk_model` exports a risk view.

For forecast analysis, use `infer_factor_premia`, `CmaDiagnosticInputs` and
`compute_cma_diagnostics` with explicit compatible units. The remaining premia,
identification, uncertainty and opportunity functions expose advanced analyses
with their own conditional assumptions. This grouping guides discovery within
the same public API. It does not change exports or imply a separate stability
guarantee. This guide targets the 0.3.0 interface; minor releases in the 0.x
series can require migration, as the [changelog](https://github.com/ArturSepp/FactorModels/blob/main/CHANGELOG.md)
and [migration guide](migration.md) record.

## `factormodels.analytics.definitions`

- `RiskFactors`
- `FactorFamily`
- `VolTargetPolicy`
- `Factor`
- `FactorModelSpec`
- `DEFAULT_VOL_WARMUP_PERIOD`
- `assemble_factor_model`
- `normalise_factor_labels`

## `factormodels.analytics.construction`

- `build_factor_panel`
- `build_factor_catalogue_panel`
- `select_factor_prices`
- `compute_factor_nav`
- `compute_risk_factor_portfolio`
- `compute_live_strategic_weights`
- `compute_volatility_targeted_portfolio`
- `DEFAULT_RETURNS_FREQ`

## `factormodels.analytics.beta_constraints`

- `compute_beta_loading_signs`

The [construction guide](construction.md) states the completed definition/panel/constraint outputs and their downstream use. Catalogue and alias mappings are supplied by the caller; the library does not select a production model.

## GLS geometry, inference and Sharpe accounting

Exported by `factormodels.gls_diagnostic`. Canonical implementations live in
`geometry`, `inference`, `resolution`, `forecast_uncertainty`, `loading_stability`,
`residual_opportunity` and `sharpe_accounting`; validation and whitening are shared
below geometry and accounting.

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

## CMA diagnostics and conditional inference

Exported by `factormodels.gls_diagnostic`; implementations live in
`cma_moments`, `cma_diagnostics`, `cma_identification` and `cma_inference`.

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

## `factormodels.analytics.models`

- `DatedFactorCovariance`
- `FactorModelSnapshot`
- `RollingFactorModel`

`FactorModelSnapshot.loadings` and `.factor_covariance` return independent labelled
copies of the fitted tables. The existing `.asset_covariance` and
`.residual_covariance` retain the selected residual policy. These accessors perform
no estimation.

## Repository teaching inputs

`synthetic_prices()` lives in `examples/reference_workflows.py`; it is excluded from the installed library.

This repository-only helper returns two
fresh deterministic DataFrames: factor NAVs and asset total-return prices.
It performs no fitting, downloading or saving. Prepare log returns with
`qis.to_returns(..., is_log_returns=True)` and pass the histories directly to
`FactorCovarEstimator`. See the [reference workflow](reference_workflows.md).

<a id="factormodels-estimation-results"></a>

## Explicit adapters and diagnostic preparation

These public operations are available from `factormodels` and
`factormodels.estimation` in 0.3.0. Their `_results` implementation module is private:

- `snapshots_from_factorlasso`
- `snapshot_from_moments`
- `estimate_historical_alpha`
- `prepare_snapshot_diagnostics`

`PreparedSnapshotDiagnostics` is exported from the root and `analytics.models`.
It contains a copied table, cutoff, alpha settings, source and availability.
Reports consume this payload explicitly. Supplied moments do not invent fit evidence.

## `factormodels.analytics.risk`

- `build_risk_model`

## Completed model contract

The [completed-output guide](completed_model.md) describes actual fields, defensive copies of
pandas contents, supported inspection/export and the exact point at which fitting
is finished. [Reference workflows](reference_workflows.md) run without providers.

`FactorCovarEstimator.fit_current` requires `cutoff` and returns a `FactorModelSnapshot`; `fit_rolling` returns `RollingFactorModel`. Fit statistics, residual histories and diagnostic fields are owned by FM; `.data` was removed in 0.3.0. The selected total/residual covariance and native-factor metadata travel with the result. Export through `.to_risk_model()` or `build_risk_model` for existing QIS consumers. The risk view does not replace the complete fit.

Current fitting clips all historical inputs before estimation and does not mutate the caller’s FL configuration. `DatedFactorCovariance` declares upstream observation/availability dates; it is rejected when unavailable at the cutoff. Supplied dates are provenance declarations, not independently verified publication evidence.

GLS/CMA results condition on their supplied loadings/error model. Their intervals do not automatically include estimated-loading uncertainty or future-return risk. Correlated residual covariance is preserved in total risk; a diagonal-reference decomposition must be explicitly requested and labelled.

## `factormodels.analytics.portfolio_context`

- `prepare_portfolio_context`

The returned dictionary completes the risk/table handoff for a downstream portfolio
result. Pass `diagnostics=prepare_snapshot_diagnostics(snapshot, alpha_span=...)`
when historical alpha is needed. Without that payload the context remains span-free.
It contains no optimiser object and performs no portfolio solve.

## `factormodels.estimation.cluster_scores`

- `extract_rolling_clusters`
- `prepare_cluster_scores`

Completed mappings and pooled score panels can be consumed independently of the
fitted model. Scoring preserves the supplied point-in-time cluster schedule.

## Opportunity and alpha uncertainty

Exported by `factormodels.gls_diagnostic`; implementations live in
`opportunity`, `alpha_uncertainty` and `universe_diagnostics`.

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

Construct with `FactorOpportunityModel.from_snapshot(snapshot, alpha)` or explicit
`betas`, raw `residual_variances`, `factor_covariance`, `alpha`, `date` and keyword-only
`residual_covariance`, `asset_covariance`, `raw_diagonal_reference_covariance`.
These matrices are prepared before GLS diagnostics; the constructor never assembles risk.
`FactorOpportunityModel.optimisation_inputs()` produces copied labelled arrays;
`model.prepare_alpha_covariance(...)` prepares the chosen objective's estimation covariance;
it is a method of `FactorOpportunityModel`, not a root-level function.
Statistical result tables remain FM outputs. Constrained portfolio solving is a
separate OP call. GLS diagnostics retain the stated diagonal-residual reference
geometry even when the selected full covariance includes residual correlation.

## Method reports

Report entry points live in `factormodels.reports`, rather than the root export list:

| Import module | Entry points and completed inputs |
|---|---|
| `factormodels.reports.covariance` | `plot_covariance_snapshot(snapshot, diagnostics=prepared)` and `run_rolling_covar_report(rolling, diagnostics=by_date)` require explicitly prepared diagnostics; neither fits a model. |
| `factormodels.reports.cma_diagnostics` | `render_cma_diagnostics(result, output_path, ...)` renders a completed `CmaDiagnosticResult`. |
| `factormodels.reports.cma_inference` | `CmaInferenceView`, `CmaInferenceReport`, `validate_cma_inference_report` and `render_cma_inference` consume completed audits and explicitly selected inference scenarios. |

The [covariance example](factor_covariance_hcgl.md) shows preparation by cutoff;
the [CMA guide](cma_diagnostics.md) describes report inputs and persistence.
Inspect exact callable signatures and full docstrings in the installed version
with `help(...)`, for example `help(FactorCovarEstimator.fit_current)` after
importing `FactorCovarEstimator` from `factormodels`.

## Local paths

`get_paths` reads YAML settings. `get_resource_path` resolves input storage;
`get_output_path` creates and resolves output storage. See the
[local path configuration](reference_workflows.md).

The root and `factormodels.utils` also export `resolve_resource_path`,
`resolve_output_path` and `create_output_directory`. Resolution does not create
directories; `create_output_directory` creates a fresh directory and rejects an
existing destination. Absolute paths bypass settings, while relative names use
the configured roots. CMA renderers share an internal atomic PDF writer that
preserves an existing report on failure and closes their generated figures.
