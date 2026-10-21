# FactorModels documentation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/tradablefactors/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

FactorModels implements multi-asset tradable factor models (MATF): construct
factor net asset value histories, estimate asset loadings and covariance, and
analyse forecasts with generalized least squares (GLS) and capital market
assumption (CMA) diagnostics. See the
[package README](https://github.com/ArturSepp/FactorModels) and
[software citation](../CITATION.cff).

Start with the [synthetic reference workflow](reference_workflows.md), then
inspect the [completed model](completed_model.md). The
[GLS/CMA guide](gls_factor_premia.md) connects the fitted model to annual arithmetic
forecasts using an explicit mean bridge and log-risk approximation. GitHub source
is the current documentation route. Committed-source links identify their stated
revision; the local guides describe the checked-out source.

```{toctree}
:maxdepth: 2

api
completed_model
reference_workflows
rolling_factor_covar_from_csv
integration
construction
factor_covariance_hcgl
gls_factor_premia
cma_diagnostics
migration
licensing
```

Start with the completed model: definitions and histories, fitted loadings, selected
covariance, diagnostics and provenance. Loading supplied inputs, fitting them,
exporting tables and consuming risk in another package are explicit separate steps.
FactorLasso supplies statistical estimation, OptimalPortfolios consumes alphas
and covariance for optimisation, and QIS supplies portfolio risk and stress
reports. The [integration guide](integration.md) describes these handoffs.
