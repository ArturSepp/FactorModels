# Direct migration to FactorModels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/tradablefactors/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Use [FactorModels](https://github.com/ArturSepp/FactorModels) directly for the
factor-model operations formerly exposed by OptimalPortfolios (OP).
FactorLasso supplies the underlying statistical objects; QuantInvestStrats (`qis`)
supplies general risk and reporting interfaces. Software citation:
[CITATION.cff](../CITATION.cff).

| Previous owner | Canonical owner |
|---|---|
| `optimalportfolios.matf.gls_diagnostic` | `factormodels.premia` |
| `optimalportfolios.matf.sharpe_accounting` | `factormodels.premia.sharpe_accounting` |
| `optimalportfolios.matf.cma_diagnostics` | `factormodels.cma.diagnostics` |
| `optimalportfolios.matf.cma_identification` | `factormodels.cma.identification` |
| `optimalportfolios.matf.cma_uncertainty` | `factormodels.cma.uncertainty` |
| `optimalportfolios.reports.cma_diagnostics` | `factormodels.reports.cma_diagnostics` |
| `optimalportfolios.reports.cma_inference` | `factormodels.reports.cma_inference` |
| OP factor covariance estimation/reporting | FactorModels estimator, fitted results and covariance reporting |
| OP comma-separated-value (CSV) factor-pipeline example and its local parser | Installed `factormodels.load_bundle`, `fit_bundle`, `save_bundle`; schema-v1 inputs and explicit actions |

Old OP exports are removed; there are no forwarding modules or compatibility extras. Update active callers and their tests. Historical experiments keep their recorded prior source/dependency environment and unchanged evidence files.

## Estimation and completed results

| Previous call or dependency | Direct replacement |
|---|---|
| OP `LassoModel`, `LassoModelType` and fitted-data aliases | Import these statistical objects from `factorlasso` |
| `fit_current_factor_covars(...)` | `FactorCovarEstimator.fit_current(..., cutoff=...)`, returning a `FactorModelSnapshot` |
| `fit_rolling_factor_covars(...)` | `FactorCovarEstimator.fit_rolling(..., time_period=...)`, returning a `RollingFactorModel` |
| Mutating estimator configuration between calls | Construct a new configuration with `dataclasses.replace`; each fit owns separate fitted state |
| Passing a fitted FactorLasso object as OP `covar_data` | Supply `factormodels.prepare_portfolio_context(snapshot)` or explicit covariance/RiskModel context |

Keep the completed FactorModels model when downstream code needs the selected covariance,
cutoff, return coordinates or provenance. Its `.data` exposes the FactorLasso container for
statistical operations that explicitly require it; that container alone does not
replace the full model contract. See the [completed-model guide](completed_model.md)
and the runnable [downstream handoff](../examples/downstream_handoff.py).

FactorModels construction accepts supplied data, factor definitions, catalogue
entries and label aliases. Data acquisition, application recipes, scheduling and
production rebalancing policy remain with the caller. See the
[construction contract](construction.md).

Current estimation requires an explicit cutoff. All supplied histories are truncated before fitting. A supplied factor covariance needs observation and availability provenance. Compare strict-cutoff results to the old reference on the same pre-truncated inputs, and record any effect on an original experiment separately.

Constrained opportunity solving remains in OP and now consumes prepared arrays.
`FactorOpportunityModel.optimisation_inputs()` supplies the constructor inputs for
`optimalportfolios.matf.OpportunityModel`. Statistical alpha covariance, including
any orthogonal projection, is prepared upstream. The solver performs no inference.

`prepare_portfolio_context(snapshot)` returns QIS RiskModel context and optional
asset diagnostic tables for `PortfolioOptimisationResult`. Full selected covariance
remains authoritative; a requested diagonal-reference decomposition is explicitly
labelled. OP accepts manually supplied covariance/RiskModel context equivalently.

`extract_rolling_clusters` and `prepare_cluster_scores` live in `factormodels.signals`.
OP retains generic scoring of supplied cluster mappings. Pass precomputed pooled
scores to downstream alpha/optimisation workflows. OP has no FactorModels or FactorLasso runtime dependency,
factor extra, root FactorLasso aliases or legacy `covar_data` constructor.

## Consumer verification and adoption

Saved-fit research reproduction loads the recorded loadings, factor covariance,
residual statistics and forecasts. Changing the package owner does not authorise
refitting those observations, replacing the input vintage or rewriting historical
receipt hashes. Compare the reproduced numerical tables and retain the original
source/dependency environment for historical checks. Current forecasts and any
intentional cutoff correction require a separate, explicitly dated comparison.

Validate downstream entry points with their own catalogue, providers, calendars,
solver and estimation policy. Compare a frozen case and the intended input date
before replacing an environment or accepted outputs.

Test an explicit cohort of built FactorModels and OP wheels in an isolated
consumer environment. Keep OP's base-wheel check separate, with neither
FactorModels nor FactorLasso installed, and require the composed checks to collect
and pass without skips. Use the declared FactorLasso dependency floor and retain
its licence metadata; see the [licensing guide](licensing.md). A local development
cohort establishes the behaviour of those artifacts, not package availability or
production adoption.
