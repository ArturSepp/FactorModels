---
myst:
  html_meta:
    description: >-
      Direct migration to factormodels for the factor-model operations formerly exposed by OptimalPortfolios: estimation, completed results and consumer verification.
---

# Direct migration to FactorModels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Use [FactorModels](https://github.com/ArturSepp/FactorModels) directly for the
factor-model operations formerly exposed by OptimalPortfolios (OP).
FactorLasso supplies the underlying statistical objects; QuantInvestStrats (`qis`)
supplies general risk and reporting interfaces. Software citation:
[CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

| Previous owner | Canonical owner |
|---|---|
| `optimalportfolios.matf.gls_diagnostic` | `factormodels.gls_diagnostic` |
| `optimalportfolios.matf.sharpe_accounting` | `factormodels.gls_diagnostic.sharpe_accounting` |
| `optimalportfolios.matf.cma_diagnostics` | `factormodels.gls_diagnostic.cma_diagnostics` |
| `optimalportfolios.matf.cma_identification` | `factormodels.gls_diagnostic.cma_identification` |
| `optimalportfolios.matf.cma_uncertainty` | `factormodels.gls_diagnostic.cma_inference` |
| `optimalportfolios.reports.cma_diagnostics` | `factormodels.reports.cma_diagnostics` |
| `optimalportfolios.reports.cma_inference` | `factormodels.reports.cma_inference` |
| OP factor covariance estimation/reporting | FactorModels estimator, fitted results and covariance reporting |
| OP comma-separated-value (CSV) factor-pipeline example and its local parser | Load DataFrames with pandas; prepare log returns with QIS; call `FactorCovarEstimator` directly |

Old OP exports are removed; there are no forwarding modules or compatibility extras. Update active callers and their tests. Historical experiments keep their recorded prior source/dependency environment and unchanged evidence files.

## Removing the input bundle API

The development version removes `factormodels.bundle`, `BundleSettings`,
`FactorBundle`, `BUNDLE_SCHEMA_VERSION`, `load_bundle`, `fit_bundle`, `save_bundle`
and `synthetic_bundle`. There is no forwarding adapter. Use plain DataFrames,
configure `LassoModel` and `FactorCovarEstimator` explicitly, then call
`fit_current` or `fit_rolling`. The repository example's `synthetic_prices()` supplies two deterministic
teaching DataFrames. The [CSV guide](rolling_factor_covar_from_csv.md) explains
input loading, currency preparation and persistence at the caller boundary.
Preserve historical six-file inputs and use their recorded environment when
reproducing old runs; this removal does not rewrite those files.

## Unreleased 0.3.0 completed-result migration

| Previous call | Replacement |
|---|---|
| `snapshot.data.y_betas`, `.x_covar`, `.y_variances` | `snapshot.loadings`, `.factor_covariance`, `.fit_statistics` |
| `snapshot.data.residuals`, `.residual_metadata` | `snapshot.residuals`, `.residual_metadata` |
| `snapshot.data.estimate_alpha(...)` | `estimate_historical_alpha(snapshot, ...)` |
| `snapshot.data.get_snapshot(...)` | `prepare_snapshot_diagnostics(snapshot, ...).table`, with selected-risk volatilities |
| `rolling.data.data` | `rolling.snapshots` |
| `rolling.data.get_y_covars()` | `rolling.covariances` |
| `rolling.data.get_y_betas()` | `{date: snapshot.loadings for date, snapshot in rolling.snapshots.items()}` |
| `FactorModelSnapshot(data=raw, cutoff=date, ...)` | `snapshots_from_factorlasso({date: raw}, residual_type=...).snapshots[date]` |
| `build_risk_model(raw, residual_type=...)` | `build_risk_model(snapshots_from_factorlasso(raw, residual_type=...))` |
| Opportunity construction from B/D/F/alpha alone | Prepare `snapshot_from_moments`, then `FactorOpportunityModel.from_snapshot(snapshot, alpha)` |
| Report fitting from prices/estimator | Fit rolling models first, prepare diagnostics per cutoff, then `run_rolling_covar_report(rolling, diagnostics=prepared)` |
| Implicit historical alpha in portfolio context | Pass `diagnostics=prepare_snapshot_diagnostics(snapshot, alpha_span=...)` explicitly |

The opportunity reference uses raw residual D, including when selected R has a
nonunit variance weight. The full selected risk remains authoritative. Root public
names remain available; these changed signatures still require caller migration.

Select the residual policy when fitting or adapting raw inputs. Completed covariance
is already assembled: passing a different policy to risk export cannot change it.
For supplied correlated moments, pass the prepared FL correlation and its existing
weight to `snapshot_from_moments(..., residual_type="empirical")` before constructing
the opportunity model. Do not replace selected full residual covariance with its
diagonal to make an older constructor work. Historical stored FL objects retain
their original source and environment; this migration does not rewrite them.

## Estimation and completed results

| Previous call or dependency | Direct replacement |
|---|---|
| OP `LassoModel`, `LassoModelType` and fitted-data aliases | Import these statistical objects from `factorlasso` |
| `fit_current_factor_covars(...)` | `FactorCovarEstimator.fit_current(..., cutoff=...)`, returning a `FactorModelSnapshot` |
| `fit_rolling_factor_covars(...)` | `FactorCovarEstimator.fit_rolling(..., time_period=...)`, returning a `RollingFactorModel` |
| Mutating estimator configuration between calls | Construct a new configuration with `dataclasses.replace`; each fit owns separate fitted state |
| Passing a fitted FactorLasso object as OP `covar_data` | Supply `factormodels.prepare_portfolio_context(snapshot)` or explicit covariance/RiskModel context |

Keep the completed FactorModels model when downstream code needs the selected covariance,
cutoff, return coordinates or provenance. The unreleased 0.3.0 candidate removes `.data`;
inspect FM-owned fields and prepare historical alpha explicitly. See the [completed-model guide](completed_model.md)
and the runnable [downstream handoff](../examples/downstream_handoff.py).

FactorModels construction accepts supplied data, factor definitions, catalogue
entries and label aliases. Data acquisition, application recipes, scheduling and
production rebalancing policy remain with the caller. See the
[construction contract](construction.md).

Current estimation requires an explicit cutoff. All supplied histories are truncated before fitting. A supplied factor covariance needs observation and availability provenance. Compare strict-cutoff results to the old reference on the same pre-truncated inputs, and record any effect on an original experiment separately.

Constrained opportunity solving remains in OP and now consumes prepared arrays.
`FactorOpportunityModel.optimisation_inputs()` supplies the constructor inputs for
`optimalportfolios.optimization.opportunity.OpportunityModel`. Statistical alpha covariance, including
any orthogonal projection, is prepared upstream. The solver performs no inference.

`prepare_portfolio_context(snapshot)` returns QIS RiskModel context and optional
asset diagnostic tables for `PortfolioOptimisationResult`. Full selected covariance
remains authoritative; a requested diagonal-reference decomposition is explicitly
labelled. OP accepts manually supplied covariance/RiskModel context equivalently.

`extract_rolling_clusters` and `prepare_cluster_scores` live in `factormodels.estimation.cluster_scores`.
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

## Named implementation modules

Package initializers export public names; implementations live in named modules.
Use the stable root imports such as `from factormodels import FactorCovarEstimator`.
Direct module imports follow these locations:

| Previous module | Implementation module |
|---|---|
| `factormodels.spec`, `factormodels.spec.definitions` | `factormodels.analytics.definitions` |
| `factormodels.construction`, `factormodels.spec.construction` | `factormodels.analytics.construction` |
| `factormodels.constraints`, `factormodels.estimation.beta_constraints` | `factormodels.analytics.beta_constraints` |
| `factormodels.model`, `factormodels.estimation.models` | `factormodels.analytics.models` |
| `factormodels.risk`, `factormodels.estimation.risk` | `factormodels.analytics.risk` |
| `factormodels.signals`, `factormodels.analytics.signals` | `factormodels.estimation.cluster_scores` |
| `factormodels.estimation` | `factormodels.estimation.estimator` (estimator export remains) |
| `factormodels.portfolio_context` | `factormodels.analytics.portfolio_context` |

The removed paths have no forwarding modules. Construction, completed contracts,
and risk/context helpers are also exported by `factormodels.analytics`; scoring
and fitting/preparation operations are exported by `factormodels.estimation`.
The stable public root names remain available. GLS/CMA and opportunity analytics
are exported by `factormodels.gls_diagnostic`; their old paths are removed.

| Removed module | Canonical implementation |
|---|---|
| `factormodels.premia._inference` | `gls_diagnostic.geometry` and `gls_diagnostic.inference` |
| `factormodels.premia._resolution` | `gls_diagnostic.resolution` |
| `factormodels.premia._scenarios` | `gls_diagnostic.forecast_uncertainty` |
| `factormodels.premia._stability` | `gls_diagnostic.loading_stability` |
| `factormodels.premia._opportunity` | `gls_diagnostic.residual_opportunity` |
| `factormodels.premia.sharpe_accounting` | `gls_diagnostic.sharpe_accounting`; projection/mimicking calls use `geometry` |
| `factormodels.cma.moments` | `gls_diagnostic.cma_moments` |
| `factormodels.cma.diagnostics` | `gls_diagnostic.cma_diagnostics` |
| `factormodels.cma.identification` | `gls_diagnostic.cma_identification` |
| `factormodels.cma.uncertainty` | `gls_diagnostic.cma_inference` |
| `factormodels.analytics.opportunity` | `gls_diagnostic.opportunity` |
| `factormodels.analytics.uncertainty` | `gls_diagnostic.alpha_uncertainty` |
| `factormodels.analytics.universe_diagnostics` | `gls_diagnostic.universe_diagnostics` |

All implementation paths in the right column are under `factormodels`.
Labelled GLS/accounting numerical inputs now require numeric pandas dtypes and
unique, nonmissing, exactly aligned labels. Object/string numerical tables must
be converted explicitly by the caller; missing labels must be resolved. Unlabelled
array/list float conversion and descriptive metadata tables retain their contracts.
See the [GLS contracts guide](gls_contracts.md) for conditioning and units.

`synthetic_prices()` is a repository example helper in `examples/reference_workflows.py`,
and is no longer an installed-library export. The README includes its input generation
so those examples also run with only the installed package. Historical saved Python
objects remain associated with their original source and environment; this relocation
does not rewrite saved fits or install compatibility packages.


## Volatility targeting belongs to the factor

`compute_factor_nav` reads the factor's configured volatility policy. The `INHERIT`
sentinel and call-time `vol_target` argument are removed in the development API.

| Intended behaviour | Factor configuration |
|---|---|
| Inherit the family target | `vol_target_policy=VolTargetPolicy.INHERIT, vol_target=None` |
| Disable targeting | `vol_target_policy=VolTargetPolicy.NONE, vol_target=None` |
| Use an explicit target | `vol_target_policy=VolTargetPolicy.FIXED, vol_target=0.10` |

Existing numeric factor overrides under `INHERIT` retain their meaning. To adapt a
research call, create a configured copy with `dataclasses.replace` and pass it as
`factor`; the original declaration stays unchanged. This also records the selected
target in the factor's configuration and text representation. `returns_freq` and
`ticker` are keyword-only, so an old positional target cannot silently bind to a
different setting.
