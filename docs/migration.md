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

Old OP exports are removed; there are no forwarding modules or compatibility extras. Update active callers and their tests. Historical experiments keep their recorded prior source/dependency environment and unchanged evidence files.

Current estimation requires an explicit cutoff. All supplied histories are truncated before fitting. A supplied factor covariance needs observation and availability provenance. Compare strict-cutoff results to the old reference on the same pre-truncated inputs, and record any effect on an original experiment separately.

Constrained opportunity solving remains in OP. At the stages 0–3 checkpoint, retained OP opportunity code still calls canonical TF numerical methods directly; stage 5 completes the supplied-input boundary and removes that coupling. No OP base-independence or production-cutover claim is made at this checkpoint.
