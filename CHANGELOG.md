# Changelog

FactorModels constructs multi-asset tradable factor models and provides
generalized least squares (GLS) and capital market assumption (CMA) diagnostics.
The versions below record pre-release development; none has been published.

## 0.1.0rc4 — unpublished FactorModels rename candidate (2026-10-20)

- Rename the distribution and import namespace from `tradablefactors` to
  `factormodels`, displayed as FactorModels (FM). No forwarding package is installed.
- Retain the public class/member names, fitted estimator outputs, construction
  settings, residual policies, numerical conventions and dependency resolutions.
- Describe factors as supplied price/level histories with explicit return
  transformations. Bundle schema versions and saved-fit input formats are unchanged.
- Adapt integrations to the new namespace while retaining the existing repository
  URL.
- Add portable contributor setup, locked source checks, installed-artifact
  verification and strict documentation build instructions.

## 0.1.0rc3 — unpublished release candidate (2026-10-19)

- Connect the fitted-model quickstart to GLS/CMA through an explicit second-order
  mean bridge, retaining annual log covariance as the declared risk approximation.
- Export `annual_arithmetic_from_log` for labelled mean and variance inputs.
  Reconcile arithmetic alpha using both asset and factor variance adjustments.
- Export fitted-model GLS/CMA tables and a PDF when the teaching assumptions and
  positive diagonal residual geometry apply; retain full model tables otherwise.
- Clarify citation wording, synthetic shrinkage and downstream use. Preserve
  fitted estimator values, seeds, selected covariance policies and dependencies.

## 0.1.0rc2 — unpublished release candidate (2026-10-19)

This candidate provides factor definitions and net asset value (NAV) construction, fitted asset
loadings and annual covariance at a stated cutoff, rolling model estimates,
GLS factor-premium inference, and CMA diagnostics with method-specific reports.
Completed model objects retain labels, conventions and provenance for downstream
risk analysis and portfolio optimisation.

Changes since `0.1.0rc1`:

- Add labelled, copy-returning `FactorModelSnapshot.loadings` and
  `factor_covariance` properties without changing estimation or covariance values.
- Install and export `synthetic_bundle()` for an offline quickstart that works
  without a source checkout. Preserve its existing seed, inputs and settings.
- Keep log-return bundle results separate from arithmetic-excess GLS/CMA reports.
  Bundle examples export fitted tables and explain why no arithmetic audit is
  produced. A separate arithmetic example demonstrates premia, diagnostics and
  explicit report export with consistent cash, alpha and covariance conventions.
- Clarify the package's completed outputs and downstream use in the README,
  and add the multi-asset tradable factor (MATF) working-paper citation and package discovery metadata.
- Retain estimator outputs, numerical library implementations and dependency
  versions.

## 0.1.0rc1 — unpublished release candidate (2026-10-19)

- Align package, citation and README version metadata without numerical or API
  changes.
- Retain the MIT source licence, supplemental notices and verified MIT FactorLasso
  dependency floor. Dependency versions and resolution remain unchanged.

## 0.1.0.dev3 — development (2026-10-18)

- Require the verified MIT FactorLasso release `>=1.0.4.post1` in base installs
  and the lock, excluding earlier GPL releases without changing estimation or
  covariance behaviour. Keep optimiser installation a separate consumer step.
- Ship supplemental OpenBLAS/LAPACK notices and document the reviewed Windows
  compiler-runtime exceptions and vendor redistributables separately from MIT source.
- Document direct TradableFactors replacements for former OptimalPortfolios
  estimator calls, FactorLasso aliases and result-context helpers; compatibility
  forwarding APIs are not provided.
- Add the installed, versioned six-file bundle API with explicit loading,
  in-memory fitting and new-directory persistence. Unsupported or unversioned
  settings are rejected; historical bundles remain in their recorded environments.
- Provide offline synthetic, supplied-bundle and public-proxy reference workflows
  with inspectable results and separate downstream QuantInvestStrats (`qis`) and
  OptimalPortfolios use.
- Document completed model fields, units, cutoffs and mutability; move factor
  estimation and GLS/CMA method documentation to its owning package.

## 0.1.0.dev2 — development (2026-10-18)

- Add opportunity GLS diagnostics, alpha-uncertainty inference and universe
  diagnostics formerly in OptimalPortfolios. Return prepared arrays for downstream
  solving.
- Add `prepare_portfolio_context` for QIS risk and optional diagnostic-table
  handoff, preserving full residual covariance and native factor coordinates.
- Add rolling cluster extraction and stability-pooled score preparation for
  downstream consumers.
- Use FactorLasso for statistical estimation and QIS for general risk/reporting
  interfaces, without introducing an optimiser dependency.

## 0.1.0.dev1 — development

- Add factor families, basket/model specifications and assembly from an explicitly supplied catalogue.
- Add basket construction, volatility targeting, composite NAVs, supported
  prehistory splicing and loading-sign constraints with explicit numerical policies.
- Accept caller-supplied label aliases and catalogues without a data-provider
  dependency.
- Add offline construction examples and independent held-unit, warm-up,
  shared-gate and splice verification.

## 0.1.0.dev0 — development

- Extract GLS, Sharpe accounting, CMA identification and conditional inference from OptimalPortfolios.
- Introduce explicit current/rolling fitted results and cutoff-aware factor covariance estimation.
- Export the selected total covariance to existing QIS risk interfaces with stated decomposition limits.
- Preserve first-party MIT attribution.
