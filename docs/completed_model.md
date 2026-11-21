---
myst:
  html_meta:
    description: >-
      The completed factormodels snapshot: labelled loadings, factor, residual and asset covariance, units, cutoff, residual policy, provenance and table export.
---

# The completed factor model

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/91f424eeabc1cd99880aef829feb6af9c61b046b)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

A completed model is bounded by its supplied universe, information cutoff and
configuration. It is usable before forecasts, optimisation or reports are added.
The [canonical offline example](https://github.com/ArturSepp/FactorModels/blob/main/examples/reference_workflows.py)
returns a fitted log-return snapshot from supplied DataFrames, then connects eligible models to
GLS/CMA reporting with an explicit second-order bridge for expected returns. The
fitted covariance is retained as a declared risk approximation. A
[separate arithmetic reference](../examples/gls_cma_workflow.py) independently
checks GLS on simple-return inputs.

## Inputs and completion points

| Producer | Explicit inputs | Completed output |
|---|---|---|
| `assemble_factor_model`, `build_factor_panel` | Catalogue, recipe, component histories, construction cadence | `FactorModelSpec` and ordered NAV/observed-level DataFrame |
| `FactorCovarEstimator.fit_current` | Factor histories, prepared asset observations in the declared coordinate, FL settings, cutoff | `FactorModelSnapshot` |
| `FactorCovarEstimator.fit_rolling` | Same information plus rolling dates | `RollingFactorModel.snapshots`, keyed by cutoff |
| `compute_cma_diagnostics` | Supplied forecasts and fitted arrays | Separate `CmaDiagnosticResult` tables; does not revise the model |

Successful return means construction or fitting has completed its validation. An
invalid input or failed fit raises; the FM snapshot has no generic `status` field.
Do not invent an `optimal` status for a statistical fit. Inspect FL diagnostics
and residual histories to assess its statistical adequacy.

## Inspectable fields and units

The completed snapshot, rolling model and prepared diagnostic containers compare and hash
by object identity. Compare their individual tables with `pandas.testing.assert_frame_equal`
when checking numerical equivalence. Their representations summarize the object rather than
printing its stored tables.

`provenance['factor_observation_date']` records the last available input price row through
the cutoff. Weekly covariance sampling may use an earlier return: for a Tuesday cutoff,
the last Wednesday return can precede that price row. Input histories need observations at
or before the cutoff, not an observation on the exact cutoff date.

| Field | Meaning and units |
|---|---|
| Caller configuration and `FactorCovarEstimator` | Recipe identity, currency/hedging preparation, cadences, transformations, loss scaling and residual policy |
| `risk_factor_prices` | Ordered NAVs or typed observed levels; LOG uses dimensionless returns, DIFFERENCE retains native points |
| `snapshot.loadings` | Independent labelled copy, assets × factors; declared decimal asset coordinate per factor return unit |
| `snapshot.factor_covariance` | Independent labelled copy of annual factor covariance in declared native coordinates |
| `snapshot.residual_covariance` | Selected annual asset residual covariance, including residual variance/correlation policy |
| `snapshot.asset_covariance` | Annual asset covariance in the declared coordinate, preserving the full selected residual block |
| `snapshot.raw_diagonal_reference_covariance` | Annual orthogonal covariance with unit raw residual-variance weight; used for GLS reference risk |
| `snapshot.fit_statistics`, `snapshot.fit_diagnostics` | Raw fit statistics and optional span-free display table; no implicit historical-alpha span |
| `snapshot.residuals`, `snapshot.residual_metadata` | Native residual history and per-asset cadence, beta span, annualisation and stored/raw scale |
| `snapshot.residual_correlation`, `snapshot.residual_correlation_metadata` | Plain dimensionless correlation plus common residual returns, metadata, frequency/span and separate observation/estimation dates |
| `snapshot.clusters`, `.linkages`, `.cutoffs`, `.derived_signs` | Available fit diagnostics; per-frequency cluster/linkage/cutoff views are materialized too |
| `snapshot.clustering_source_assets` | Original ordered dendrogram leaf identities by cadence; retained when selecting a smaller universe |
| `snapshot.residual_cluster_targets`, `.residual_target_source_assets` | Full-fit block correlations on the current asset axis, and the original target universe when known |
| `snapshot.cutoff`, `snapshot.provenance` | Information cutoff, latest observations, conventions and producer settings; retain caller-specific recipe, currency, units and source evidence separately |

Factor and asset labels and their ordering are substantive. Monthly covariance is
annualised with 12 observations per year; each supported cadence uses the existing
QIS/FL annualisation. Annual covariance contains variance units, not percent
volatility. Supplied asset observations are LOG by default; total/excess basis must
agree with the factor basis. Explicit `asset_return_type="simple_payoff"` accepts
prepared arithmetic payoff increments on one common factor/response/residual
cadence and records that coordinate in provenance. It performs no compounding
or logarithm of a signed payoff. See [Equity FX payoff risk](equity_fx_risk.md). Input timestamps declare availability; the estimator
cannot discover provider publication delays or restatements.

The synthetic example produces two factors, four assets and a 4 × 4 annual asset
covariance at **2024-01-31**, using 121 monthly input level observations. It verifies
the independent identity $\Sigma = B F B^{\mathsf T} + R$ with explicit component sums.
The construction-only example retains its separate model specification; a loaded
CSV panel alone does not retain an executable reconstruction of its recipe.

## Ownership, mutability and persistence

From **0.3.0**, completed FM results own plain data and retain
no FL containers. All public pandas/NumPy fields and nested metadata are defensive
copies. Rolling map retrieval returns a new map of immutable completed snapshots.
Inspection and `to_risk_model()` perform neither fitting nor FL covariance assembly.
There is no `.data` view. Prepare another snapshot to change its residual policy;
changing a policy label alone cannot reassemble stored covariance.
Use `snapshot.with_provenance({...})` to attach caller currency, units or source
declarations while retaining the same completed moments and independent metadata.

Historical alpha is an explicit estimation operation:

```python
from factormodels import estimate_historical_alpha, prepare_snapshot_diagnostics
from factormodels import prepare_portfolio_context

alpha = estimate_historical_alpha(snapshot, alpha_span={"ME": 120, "QE": 40})
diagnostics = prepare_snapshot_diagnostics(snapshot, alpha_span={"ME": 120, "QE": 40})
context = prepare_portfolio_context(snapshot, diagnostics=diagnostics)
```

Scalar spans and frequency maps retain FL's existing missing-observation and scaling
semantics. Without history, full fit diagnostics retain the in-sample-alpha fallback.
Minimal supplied moments do not invent history, R-squared or in-sample alpha:
prepared diagnostics mark alpha unavailable, while direct alpha estimation raises.
Report consumers verify the prepared cutoff and asset order and do not choose a span.
Omitting diagnostics from portfolio context returns available span-free fields only.

Use `snapshots_from_factorlasso(raw, residual_type="empirical")` for existing raw FL
fits (a dated `RollingFactorModel` is returned even for a single raw fit), or
`snapshot_from_moments(betas, factor_covariance, residual_variances, date=cutoff)`
for supplied moments. Both prepare risk through FL before returning FM-owned fields.
The raw diagonal reference is distinct from `to_risk_model(diagonal_reference=True)`,
whose decomposition uses the diagonal of selected R and retains full selected Sigma.

Input loading and artifact publication belong to the caller. From **0.6.0**,
`model_to_payload(model)` and `model_from_payload(payload)` provide a
strict schema-1 contract for the completed state. It contains independent copies
of every pandas/NumPy field, axes, cutoff, selected policy, original clustering
evidence and provenance. Loading validates the constructor contract without
fitting, covariance assembly, dynamic class imports or fallback to FL containers.
Missing/extra fields, unsupported schema versions and duplicate rolling dates fail.
The payload is a mapping of typed fields, not a JSON string or workbook serializer.
The application must preserve index types/names, ordering, dtypes, missing-value
masks and nested metadata in its chosen lossless encoding.

`digest = write_model_archive(path, model)` and
`read_model_archive(path, expected_sha256=digest)` provide FM's
shared lossless archive encoding. Writes default to schema 1. Once every consumer
supports schema 2, use `write_model_archive(path, model, schema=2)` to store each
homogeneous numeric frame as a block and share the axes of mixed-type frames.
Both archive formats carry the same version 1 fitted-model payload. The ZIP contains
typed JSON and NumPy arrays. The reader verifies the required whole-file digest before decoding. It uses
neither pickle nor dynamic Python class imports.
Keep the complete archive as the reload source. Excel/CSV tables are reporting
views and can round numbers or lose index metadata. Applications own file naming,
the trusted digest, multi-artifact publication and any accompanying workbook.
See the [archive API](api.md) for the supported
types and validation contract.

Export individual result tables with pandas `to_csv` for inspection and retain
units, date, recipe and source provenance beside them. The example's
`persist_example` writes inspectable model tables and
`result_metadata.json` with conventions/provenance. Eligible models additionally
produce forecast tables, GLS/CMA results and a PDF. Other models retain their full
tables and an explicit unavailable-report reason. Those output CSVs are not a
fitted-object reload format.

## Native selection and composition

The [offline native-result example](../examples/native_results.py) uses supplied
moments to verify selection, causal date access, composition and payload reload.
It does not fetch data or call an optimiser:

```console
python examples/native_results.py
```

`snapshot.select_assets(["asset_b", "asset_a"])` slices every asset-indexed field
in that order. Selected covariance, residual covariance and the raw diagonal
reference are exact principal submatrices; none is recomputed. Empty, duplicate
or unknown assets fail. Explicit `rename_assets(mapping)` relabels asset axes and
retained clustering/target identities without recomputing risk; it rejects unknown
keys and duplicate resulting names. The corresponding rolling
method requires coverage at every fitted date. `rolling.at_or_before(date)`
returns the latest fit no later than the requested date, without changing its
cutoff; requesting a date before the first fit fails.

Trees and cutoffs remain evidence about the original fitted universe. Selection
retains their original leaf identities and never re-cuts a tree. Historical
filtered FL inputs may lack those identities; their target values can still be
materialized, but further tree-bearing native selection requires the missing
identity evidence rather than guessing it from the smaller universe.

`compose_snapshots(blocks, assets=order, loadings=overrides, clusters=memberships,
linkage_source=0)` combines disjoint orthogonal blocks through FL's covariance
assembly. Blocks must share the cutoff, exact factor order/covariance, native
coordinates, declared currency/units and residual variance weight. Common factor
exposure creates covariance between blocks; concatenating block-diagonal asset
covariances is incorrect. Loadings and memberships are complete explicit caller
inputs: FM supplies no cash-zeroing rule or synthetic cash label. The selected
main tree keeps its original leaf universe; all block provenance and clustering
evidence remain recorded. Blocks with prepared correlated-residual evidence are
rejected until a cross-block residual model is explicitly supported. Supplied
moments do not acquire invented alpha, R-squared or residual histories.

## Explicit residual variants

Ordinary consumers use the fitted result's selected covariance directly.
`prepare_residual_variant(model, residual_type="orthogonal")` is an explicit
research/stress operation returning another completed result. Variance and
correlation weights are arguments, not mutations of the original result.
FL remains the owner of covariance assembly. Fit statistics, diagnostic columns,
native residual cadence/scaling and the raw unit-weight diagonal reference retain
their original meanings.

Where correlation is available, materialization retains FL's full-universe
exposure/residual cluster targets before selection. Subsetting and schema reload
preserve those targets; requesting an unavailable retained target fails instead
of estimating a new target on fewer assets. An already-filtered historical FL
input can retain valid target values while its original asset identities remain
explicitly unknown.

`allow_orthogonal_burn_in=True` is an explicit replay declaration for missing
correlation before its first available estimate. A later missing estimate fails.
The result records both the target policy and whether a dated snapshot used this
fallback. Fresh fitting has no new fallback. `to_risk_model(diagonal_reference=True)`
still preserves selected total risk; it does not replace a genuinely orthogonal
variant requested by a stress calculation.

## Native lineage and descriptive statistics

`prepare_cluster_lineage(rolling)` prepares FL's statistical lineage report from
native snapshots. `prepare_cluster_fingerprints(snapshot)` returns owned plain
fingerprint fields for matching experiments. Both preserve raw residual-variance
statistics independently of the selected portfolio covariance policy. Neither
operation refits the model or returns FL fitted containers to the application.
Their full-history matching is descriptive; trading signals use the causal
memberships available at each fitted cutoff. Keep that distinction when plotting
stable labels alongside a backtest.

## Interpretation and limitations

Validation follows the receiving operation's numerical contract. Completed-model
matrices use symmetry tolerances `rtol=1e-10`, `atol=1e-12`; their factor-covariance
PSD check allows a negative eigenvalue of at most `1e-10 * max(1, max(abs(C)))`.
Prepared correlation uses an absolute eigenvalue allowance of `1e-8`. Selected
assembled asset/residual blocks are checked for labels, finiteness and symmetry;
the constructor does not reassemble them or assert a new PSD guarantee.

Equity/FX joint moments use a scale-relative eigenvalue allowance of
`1e-10 * max(max(abs(C)), 1e-12)` and symmetry `rtol=1e-10`, `atol=1e-14`.
GLS residual covariance instead requires a successful Cholesky factor and symmetry
`rtol=0`, `atol=1e-12`: its inverse geometry requires positive definiteness.
These checks validate different objects; none silently repairs a supplied matrix.

Native-point factors can represent observed levels rather than investable NAVs.
Interpreting loadings as executable hedges also requires currency, funding and
scaling assumptions. Statistical diagnostics are conditional on the chosen factor
model and supplied forecasts; a completed fit does not establish their economic
adequacy.

The canonical report adapter requires LOG factors, a zero cash forecast assumption and
strictly positive diagonal selected residual covariance. It retains the fitted
loadings and covariance while using a second-order mean bridge to construct
annual arithmetic forecasts. This is neither exact compounding nor an exact
lognormal conversion. The [GLS methodology](gls_factor_premia.md) explains the
asset-level variance adjustment and credited arithmetic alpha.

Native-point factors, correlated or zero residual covariance, and nonzero cash
remain valid fitted-model cases. The example declines their reports because it
does not supply the additional convention or metric choices. It never replaces
their selected covariance with a diagonal approximation to make them eligible.

## Separate downstream usage

`to_risk_model()` exports the selected total covariance to existing QIS risk
methods. It does not include all FL diagnostics or full residual history. With
correlated residuals, the authoritative total risk and the optional explicitly
labelled diagonal-reference decomposition have different meanings.

[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) receives supplied
alphas and covariance; [QIS](https://github.com/ArturSepp/QuantInvestStrats) receives
risk/portfolio/scenario inputs. FM's own GLS/CMA/covariance reports remain in FM and
use existing QIS plotting primitives. See the [handoff example](integration.md).
