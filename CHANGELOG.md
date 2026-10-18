# Changelog

TradableFactors owns factor construction, estimation, GLS/CMA diagnostics and their method-specific reports. Extracted consumer APIs are removed in the coordinated migration; callers use the canonical TradableFactors API directly.

## 0.1.0.dev1 — development

- Add factor families, basket/model specifications and assembly from an explicitly supplied catalogue.
- Move basket construction, volatility targeting, composite NAVs, supported prehistory splicing and loading-sign constraints to TF, preserving the existing numerical policies.
- Inject label aliases; keep private catalogues, alias maps, providers and production parameters in Rosaa.
- Migrate Rosaa callers directly and remove the former generic modules without forwarding APIs.
- Add offline construction examples and independent held-unit, warm-up, shared-gate and splice verification. This remains an unpublished migration checkpoint; rights and dependency-licensing gates still apply.

## 0.1.0.dev0 — development

- Extract GLS, Sharpe accounting, CMA identification and conditional inference from OptimalPortfolios.
- Introduce explicit current/rolling fitted results and cutoff-aware factor covariance estimation.
- Export the selected total covariance to existing QIS risk interfaces with stated decomposition limits.
- Preserve first-party MIT attribution. Public release remains subject to the verified MIT FactorLasso dependency floor and the remaining migration gates.
