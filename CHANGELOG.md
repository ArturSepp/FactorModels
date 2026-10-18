# Changelog

TradableFactors owns factor estimation, GLS/CMA diagnostics and their method-specific reports. Extracted OptimalPortfolios APIs are removed in the coordinated migration; callers use the canonical TradableFactors API directly.

## 0.1.0.dev0 — development

- Extract GLS, Sharpe accounting, CMA identification and conditional inference from OptimalPortfolios.
- Introduce explicit current/rolling fitted results and cutoff-aware factor covariance estimation.
- Export the selected total covariance to existing QIS risk interfaces with stated decomposition limits.
- Preserve first-party MIT attribution. Public release remains subject to the verified MIT FactorLasso dependency floor and the remaining migration gates.
