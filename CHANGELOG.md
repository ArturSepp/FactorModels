# Changelog

FactorModels constructs multi-asset tradable factor models and provides
generalized least squares (GLS) and capital market assumption (CMA) diagnostics.
The first public release is 0.1.0; earlier entries record unpublished development.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/): a behaviour
change leads each release, then `Added`, `Changed`, `Fixed`, `Removed`.

## [0.6.1] - 2026-11-21

`read_model_archive` accepts archive schemas 1 and 2. Existing calls to
`write_model_archive(path, model)` produce schema 1 archives readable by FM 0.6.0.

### Added

- Pass `schema=2` to store homogeneous numeric frames as single arrays and
  write the axes once per frame. Deploy a compatible reader to every consumer
  before opting in. The fitted-model payload remains at version 1.

## [0.6.0] - 2026-11-19

Completed results support native selection, composition, persistence and lineage
without reconstructing fitted FactorLasso containers in downstream applications.

### Added

- `prepare_cluster_lineage` and `prepare_cluster_fingerprints` prepare descriptive
  statistical reports and plain fingerprint fields from native completed results.
  They preserve FL's existing algorithms and raw residual-risk convention;
  applications no longer need fitted FL containers for lineage research.

- `write_model_archive` and `read_model_archive` share a lossless, checksummed
  native-result archive across independent consumers. The schema preserves typed
  fields and categorical vocabularies without pickle, fitting or risk assembly.
- `FactorModelSnapshot.rename_assets` explicitly relabels all fitted axes and
  retained tree identities without recomputing risk or changing leaf order.

- `prepare_cluster_labels` returns plain descriptive lineage labels for completed
  FM results using FL's existing MCF classification and matching defaults.

- Select completed assets in caller order with `select_assets`, retaining exact
  covariance submatrices, original dendrogram identities and full-fit residual
  cluster targets. `RollingFactorModel.at_or_before` resolves causal dated access.
- Compose compatible orthogonal snapshots with `compose_snapshots`, preserving
  cross-block factor covariance. Applications explicitly supply cash loadings,
  synthetic memberships and the retained main linkage source.
- Prepare explicit research/stress residual policies with `prepare_residual_variant`,
  using FL assembly and retained targets. Historical orthogonal burn-in is opt-in;
  a missing residual correlation after the first available estimate remains an error.
- Encode/decode complete results with `model_to_payload` and `model_from_payload`.
  The strict versioned plain-field schema retains numerical tables and evidence;
  application file formats and workbook policies remain outside FM.

### Fixed

- Retain original clustering tree leaves for warm-up assets with unavailable
  memberships when the producer supplies their cadence. Selection keeps their
  identities without inventing assignments or inferring missing historical evidence.

## [0.5.1] - 2026-11-17

### Fixed

- Completed model, diagnostic, dated covariance and equity-risk containers use identity
  equality and hashing. Comparing distinct equal-valued tables no longer raises pandas'
  ambiguous-truth error. Representations omit full numerical tables.
- `RiskFactors.validate_columns` accepts one-shot iterables without consuming them repeatedly.
- Volatility-targeted construction omits rebalance labels beyond the last supplied price.
  Historical weights and NAV are preserved; returned weights no longer include unusable
  trailing rows. Valid non-trading period-end labels remain available for execution.
- `Factor.initial_volatility` correctly declares its nullable return type.

### Changed

- Refresh the reviewed OP consumer and public handbook handoffs. Keep external link-health
  checks in their owning documentation workflows.
- Render the complete root API with parameter documentation and expand the construction
  methodology. Cite the accepted CMA paper as forthcoming in JPM (2027).
- Examples expose named `LocalTests` cases while retaining existing integer calls.
- Enforce E/W style rules alongside the existing annotation and import boundaries.

### Removed

- Retire migration replay tools and unused private estimator compatibility methods. Keep
  independent numerical contracts in the shipped component tests and protect private papers
  in both the Git index and built distributions.

## [0.5.0] - 2026-11-08

### Added

- Add `EquityPayoffRiskSpec`, `EquityPayoffRiskSnapshot`, `build_funded_currency_exposure`,
  `compute_payoff_factor_returns`, `assemble_equity_payoff_risk`, `EquityFxRiskSpec`,
  `EquityFxRiskSnapshot`, `build_currency_exposure` and `augment_equity_fx_model`.
  `FactorCovarEstimator.asset_return_type` declares log returns or single-cadence SIMPLE payoffs.
- Add research-only SIMPLE equity payoff risk with arithmetic basket
  factors, cash-aware opening currency exposure, and every factor/residual/FX
  cross block retained. Separate fixed-exposure forecasts from dated historical
  reconstruction; no production activation is included.

- Add equity/FX risk augmentation of a completed local log-excess model,
  with native factor coordinates, dated structural exposures, full residual/FX
  covariance, and QIS risk interchange. The supplied local covariance is preserved
  by default only when explicitly selected; joint re-estimation is a separate policy.
- Reject incompatible preserved cross blocks unless explicit, recorded scalar
  attenuation is selected. Preserve the full selected residual covariance.
- Add offline covariance references and an executable multi-currency example.
  This first-order risk prototype does not implement self-financed FX accounting
  or establish accuracy on an observed equity universe. No existing fit defaults change.

### Fixed

- Update the handbook for the released 0.3.0 completed-model interface: FM-owned
  diagnostics, selected residual policy, explicit report preparation and current
  GLS module paths. Make the covariance guide's excerpts executable in sequence.
- Document released FM 0.3.0 / OP 9.0.1 requirements for the optional solver handoff,
  distinguish installed APIs from checkout examples, and show the documentation
  build version with links to the stable and development channels.
- Restore complete docstring coverage for existing validation and report-lifecycle
  tests without changing their calculations or assertions.

## [0.3.0] - 2026-10-25

This breaking release removes snapshot/rolling `.data` views. Read completed
FM fields; pass raw FL fits through `snapshots_from_factorlasso` before risk export.
Opportunity constructors consume prepared snapshots or complete covariance views,
and reports require explicitly prepared historical-alpha diagnostics. Retired
`spec`, `premia`, `cma` and moved deep imports have no forwarding modules. Follow
the [migration guide](docs/migration.md) when upgrading callers from 0.2.0.

### Changed

- Split private estimation orchestration into cadence fitting, cluster scheduling
  and residual preparation/reuse stages, preserving numerical operation order.
- Complete production signature annotations and mathematical ownership headers;
  remove all temporary production annotation/header exemptions.
- Share CMA report PDF persistence, preserving output resolution, atomic replacement,
  Matplotlib style restoration, figure closure and cleanup on failure. Use verified
  public QIS imports for factor returns, portfolio variance and return conventions.
- GLS, CMA and opportunity/uncertainty analytics now share `factormodels.gls_diagnostic`.
  Geometry and premium inference have separate modules; shared validation and
  Cholesky whitening sit below geometry and Sharpe accounting. Accounting uses the
  full whitened design and does not inherit geometry's retained-rank truncation.
- GLS/accounting labelled numerical inputs now reject missing axis labels and
  nonnumeric pandas dtypes, including object/string data that could previously be
  coerced to floats. Both covariance axes must align exactly, including when B is
  unlabelled. Unlabelled array/list conversion and descriptive metadata tables retain
  their contracts. Existing valid numerical results and established error text remain.
- Shipped tests now use component-adjacent `tests/test_*.py` locations and one discovery
  convention; the installed-wheel CI verifies each component and nonempty collection.
- Factor definitions, construction, beta constraints, completed model contracts and
  risk export now live in `factormodels.analytics`; cluster scoring lives in
  `factormodels.estimation.cluster_scores`. Public root names and numerical results
  are preserved. The migration guide lists the removed deep paths.
- Completed snapshots own plain FM fields, including selected residual/asset covariance,
  raw unit-weight diagonal-reference covariance, fit statistics, histories and separate
  correlation observation/estimation dates. Mutable fields return defensive copies.
- Risk export accepts completed FM models or plain covariance. Convert raw FL results
  explicitly with `snapshots_from_factorlasso`; use `snapshot_from_moments` for supplied
  moments. Removed risk-policy keywords raise `TypeError`.
- Opportunity diagnostics consume completed snapshots through `from_snapshot`, or explicit
  full moments and all three prepared covariance views. GLS retains raw residual D.
- Covariance reports consume completed rolling models and dated prepared diagnostics.
  Portfolio context accepts explicit prepared diagnostics; its default is span-free.

### Added

- `resolve_resource_path`, `resolve_output_path` and `create_output_directory` are
  exported by `factormodels.utils` and the root package with unchanged path semantics.
- Dependency-direction checks and zero-debt annotation/header inventories prevent
  regressions in the completed production contracts.
- `estimate_historical_alpha`, `prepare_snapshot_diagnostics` and
  `PreparedSnapshotDiagnostics` preserve FL alpha semantics through explicit preparation,
  including the in-sample fallback and explicit unavailability for minimal moments.

### Removed

- `factormodels.premia`, `factormodels.cma`, and analytics opportunity/uncertainty/universe
  deep paths. Import their names from the root or `gls_diagnostic`. Removed the four
  geometry delegates and duplicate loadings preparation in Sharpe accounting;
  callers use `gls_diagnostic.geometry` directly. No compatibility facades remain.
- `factormodels.spec`, `estimation.models`, `estimation.risk`,
  `estimation.beta_constraints` and `analytics.signals` deep paths; no forwarding modules.
- Snapshot and rolling `.data` FL-container views. Use the named completed fields and
  preparation APIs; no retired interface forwards calls. This is an unpublished candidate.

## [0.2.0] - 2026-10-24

### Removed

- Remove the six-file input bundle layer and its public exports: `factormodels.bundle`,
  `BUNDLE_SCHEMA_VERSION`, `BundleSettings`, `FactorBundle`, `load_bundle`, `fit_bundle`,
  `save_bundle` and `synthetic_bundle`. Existing callers must prepare DataFrames and
  call `FactorCovarEstimator.fit_current` or `.fit_rolling` directly.

### Changed

- Read volatility targeting directly from `Factor.vol_target_policy` and `Factor.vol_target`.
  Remove the custom `INHERIT` sentinel and `compute_factor_nav(..., vol_target=...)` override;
  inherit, disabled and fixed targeting remain explicit factor configurations. Research callers
  use `dataclasses.replace`; `returns_freq` and `ticker` are keyword-only to reject old
  positional overrides. Construction outputs and inherited defaults are unchanged.

- Move implementations out of package initializers into named modules: definitions and
  construction under `spec`, estimator/results/risk and beta constraints under `estimation`,
  portfolio context and cluster signals under `analytics`. Initializers contain exports only.
  Remove the redundant `construction`, `constraints`, `model`, `risk`, `portfolio_context`,
  `signals` and `examples` packages. Public numerical root exports remain available.
- Keep synthetic teaching inputs in the repository examples and the self-contained README,
  removing `synthetic_prices` from the installed API. Update callers and package checks.
  Algorithms, seeds, conventions and historical environments are unchanged.

- Update the optional OP handoff to `optimalportfolios.optimization.opportunity`.
  This development example requires the matching OP development candidate.

- Examples and README configure the estimator explicitly. `synthetic_prices()` returns
  plain factor NAV and asset price DataFrames with the existing seed and histories.
- The reference CLI accepts `--factor-prices` and `--asset-prices` instead of `--bundle`.
  YAML resource/output paths and explicit result export remain supported. Currency/hedging
  preparation and input persistence belong to the caller. Estimator algorithms are unchanged.
- Update the affected source/wheel tests, CI import checks and input documentation.
  Historical environments and saved inputs remain unchanged.

## [0.1.1] - 2026-10-22

- Move root implementation modules into import-compatible subpackages; only `__init__.py`
  remains as Python source directly under `factormodels/`. Public imports and estimator outputs are unchanged.
- Resolve relative bundle, example and report paths through YAML resource/output settings,
  with explicit environment overrides. Absolute caller paths retain their destination;
  bundle and example exports still refuse to overwrite an existing directory.
- Rename the complete arithmetic GLS/CMA workflow to `examples/gls_cma_workflow.py`,
  distinct from the GLS-only demonstration in `examples/methods/gls_factor_premia.py`.

- Add a minimal model definition and estimation example with rolling covariance retrieval,
  presented as the README starting example.

## [0.1.0] - 2026-10-21

First public release. The Python floor rises from 3.10 to 3.11 and Pillow is pinned at
`>=13.0.0`, to exclude the known Pillow vulnerabilities whose fixed release no longer supports
Python 3.10; both floors are enforced in the installed package metadata and checked in CI.
Estimator source, defaults and numerical conventions are unchanged from 0.1.0rc4.

### Added

- The canonical `factormodels` distribution on PyPI and the public FactorModels source.
- Factor construction, fitted asset loadings and covariance (`FactorCovarEstimator`,
  `FactorModelSnapshot`), rolling models, GLS/CMA analytics (`infer_factor_premia`,
  `compute_cma_diagnostics`) and the explicit QIS/OptimalPortfolios handoffs.
- Public OSS consumer verification and PyPI installation instructions.

### Changed

- `requires-python` is `>=3.11`; `pillow>=13.0.0` is a declared floor.
- The verified candidate algorithms, estimator outputs and numerical defaults are retained.

## [0.1.0rc4] - 2026-10-20 (unpublished FactorModels rename candidate)

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

## [0.1.0rc3] - 2026-10-19 (unpublished release candidate)

- Connect the fitted-model quickstart to GLS/CMA through an explicit second-order
  mean bridge, retaining annual log covariance as the declared risk approximation.
- Export `annual_arithmetic_from_log` for labelled mean and variance inputs.
  Reconcile arithmetic alpha using both asset and factor variance adjustments.
- Export fitted-model GLS/CMA tables and a PDF when the teaching assumptions and
  positive diagonal residual geometry apply; retain full model tables otherwise.
- Clarify citation wording, synthetic shrinkage and downstream use. Preserve
  fitted estimator values, seeds, selected covariance policies and dependencies.

## [0.1.0rc2] - 2026-10-19 (unpublished release candidate)

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

## [0.1.0rc1] - 2026-10-19 (unpublished release candidate)

- Align package, citation and README version metadata without numerical or API
  changes.
- Retain the MIT source licence, supplemental notices and verified MIT FactorLasso
  dependency floor. Dependency versions and resolution remain unchanged.

## [0.1.0.dev3] - 2026-10-18 (development)

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

## [0.1.0.dev2] - 2026-10-18 (development)

- Add opportunity GLS diagnostics, alpha-uncertainty inference and universe
  diagnostics formerly in OptimalPortfolios. Return prepared arrays for downstream
  solving.
- Add `prepare_portfolio_context` for QIS risk and optional diagnostic-table
  handoff, preserving full residual covariance and native factor coordinates.
- Add rolling cluster extraction and stability-pooled score preparation for
  downstream consumers.
- Use FactorLasso for statistical estimation and QIS for general risk/reporting
  interfaces, without introducing an optimiser dependency.

## [0.1.0.dev1] (development)

- Add factor families, basket/model specifications and assembly from an explicitly supplied catalogue.
- Add basket construction, volatility targeting, composite NAVs, supported
  prehistory splicing and loading-sign constraints with explicit numerical policies.
- Accept caller-supplied label aliases and catalogues without a data-provider
  dependency.
- Add offline construction examples and independent held-unit, warm-up,
  shared-gate and splice verification.

## [0.1.0.dev0] (development)

- Extract GLS, Sharpe accounting, CMA identification and conditional inference from OptimalPortfolios.
- Introduce explicit current/rolling fitted results and cutoff-aware factor covariance estimation.
- Export the selected total covariance to existing QIS risk interfaces with stated decomposition limits.
- Preserve first-party MIT attribution.
