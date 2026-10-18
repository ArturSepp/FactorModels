# Direct migration to TradableFactors

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Use the [TradableFactors package](https://github.com/ArturSepp/tradablefactors) directly. Software citation: [CITATION.cff](../CITATION.cff).

| Previous owner | Canonical owner |
|---|---|
| `optimalportfolios.matf.gls_diagnostic` | `tradablefactors.premia` |
| `optimalportfolios.matf.sharpe_accounting` | `tradablefactors.premia.sharpe_accounting` |
| `optimalportfolios.matf.cma_diagnostics` | `tradablefactors.cma.diagnostics` |
| `optimalportfolios.matf.cma_identification` | `tradablefactors.cma.identification` |
| `optimalportfolios.matf.cma_uncertainty` | `tradablefactors.cma.uncertainty` |
| `optimalportfolios.reports.cma_diagnostics` | `tradablefactors.reports.cma_diagnostics` |
| `optimalportfolios.reports.cma_inference` | `tradablefactors.reports.cma_inference` |
| OP factor covariance estimation/reporting | TF estimator, fitted results and covariance reporting |
| OP CSV factor-pipeline example and its local parser | Installed `tradablefactors.load_bundle`, `fit_bundle`, `save_bundle`; schema-v1 inputs and explicit actions |
| Rosaa generic factor specifications and assembly | `tradablefactors.spec` with an injected catalogue |
| Rosaa basket/NAV construction | `tradablefactors.construction` |
| Rosaa generic loading-sign constraints | `tradablefactors.constraints` |
| Rosaa label-normalisation mechanics | `tradablefactors.normalise_factor_labels` with injected aliases |

Old OP exports are removed; there are no forwarding modules or compatibility extras. Update active callers and their tests. Historical experiments keep their recorded prior source/dependency environment and unchanged evidence files.

The former Rosaa generic specification, construction and constraint modules are also removed. Private assembly and label bindings remain where they supply Rosaa's catalogue or alias policy. Data loading, recipes, paths, scheduling and explicit production rebalancing parameters remain in Rosaa. TF construction accepts supplied data and definitions and works independently of that consumer; see the [construction contract](construction.md).

Current estimation requires an explicit cutoff. All supplied histories are truncated before fitting. A supplied factor covariance needs observation and availability provenance. Compare strict-cutoff results to the old reference on the same pre-truncated inputs, and record any effect on an original experiment separately.

Constrained opportunity solving remains in OP and now consumes prepared arrays.
`FactorOpportunityModel.optimisation_inputs()` supplies the constructor inputs for
`optimalportfolios.matf.OpportunityModel`. Statistical alpha covariance, including
any orthogonal projection, is prepared upstream. The solver performs no inference.

`prepare_portfolio_context(snapshot)` returns QIS RiskModel context and optional
asset diagnostic tables for `PortfolioOptimisationResult`. Full selected covariance
remains authoritative; a requested diagonal-reference decomposition is explicitly
labelled. OP accepts manually supplied covariance/RiskModel context equivalently.

`extract_rolling_clusters` and `prepare_cluster_scores` live in `tradablefactors.signals`.
OP retains generic scoring of supplied cluster mappings. Pass precomputed pooled
scores to downstream alpha/optimisation workflows. OP has no TF/FL runtime dependency,
factor extra, root FactorLasso aliases or legacy `covar_data` constructor.
Rosaa and active paper production cutover remains a separate stage-7 gate.
