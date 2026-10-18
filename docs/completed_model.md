# The completed factor model

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [TradableFactors](https://github.com/ArturSepp/tradablefactors).
Software citation: [CITATION.cff](https://github.com/ArturSepp/tradablefactors/blob/main/CITATION.cff).

A completed model is bounded by its supplied universe, information cutoff and
configuration. It is usable before forecasts, optimisation or reports are added.
The [canonical offline example](https://github.com/ArturSepp/tradablefactors/blob/main/examples/reference_workflows.py)
returns a bundle, fitted snapshot and separate illustrative CMA audit.

## Inputs and completion points

| Producer | Explicit inputs | Completed output |
|---|---|---|
| `assemble_factor_model`, `build_factor_panel` | Catalogue, recipe, component histories, construction cadence | `FactorModelSpec` and ordered NAV/observed-level DataFrame |
| `load_bundle` | Six schema-v1 CSVs | `FactorBundle`: validated input histories, metadata, settings and source hashes; no fit |
| `FactorCovarEstimator.fit_current` / `fit_bundle` | Factor histories, log asset returns or supplied prices/FX/rates, FL settings, cutoff | `FactorModelSnapshot` |
| `fit_rolling` / `fit_bundle(..., rolling=True)` | Same information plus rolling dates | `RollingFactorModel.snapshots`, keyed by cutoff |
| `compute_cma_diagnostics` | Supplied forecasts and fitted arrays | Separate `CmaDiagnosticResult` tables; does not revise the model |

Successful return means construction or fitting has completed its validation. An
invalid input or failed fit raises; the TF snapshot has no generic `status` field.
Do not invent an `optimal` status for a statistical fit. Inspect FL diagnostics
and residual histories to assess its statistical adequacy.

## Inspectable fields and units

| Field | Meaning and units |
|---|---|
| `bundle.settings`, `bundle.asset_metadata` | Recipe identity, currency, hedge ratios, cadences, transformations, loss scaling and residual policy |
| `bundle.factor_levels` | Ordered NAVs or typed observed levels; LOG uses dimensionless returns, DIFFERENCE retains native points |
| `snapshot.data.y_betas` | Assets × factors; decimal asset log return per factor return unit |
| `snapshot.data.x_covar` | Annual factor covariance in declared native coordinates |
| `snapshot.residual_covariance` | Selected annual asset residual covariance, including residual variance/correlation policy |
| `snapshot.asset_covariance` | Annual asset log-return covariance, preserving the full selected residual block |
| `snapshot.data` | FL `CurrentFactorCovarData`: loadings, covariance, variances, residuals and available cluster/fit diagnostics; not the mutable `LassoModel` estimator itself |
| `snapshot.cutoff`, `snapshot.provenance` | Information cutoff, latest observations, conventions and producer settings; bundle fits add recipe, currency, units and input hashes |

Factor and asset labels and their ordering are substantive. Monthly covariance is
annualised with 12 observations per year; each supported cadence uses the existing
QIS/FL annualisation. Annual covariance contains variance units, not percent
volatility. Bundle asset returns are explicitly log; total/excess basis must agree
with the declared factor basis. Input timestamps declare availability; the loader
cannot discover provider publication delays or restatements.

The synthetic example produces two factors, four assets and a 4 × 4 annual asset
covariance at **2024-01-31**, using 121 monthly input level observations. It verifies
the independent identity $\Sigma = B F B^{\mathsf T} + R$ with explicit component sums.
The construction-only example retains its separate model specification; a loaded
CSV panel carries recipe identity, not an executable reconstruction of that recipe.

## Ownership, mutability and persistence

Configuration dataclasses are frozen. `FactorModelSnapshot` and `FactorBundle`
constructors copy supplied model data; rolling containers retain their snapshots.
Settings mappings are read-only. Nested pandas tables in completed results remain
mutable. Treat them as result data: use `.copy()` for analysis and construct a new
result for a revised model. Freezing a dataclass does not make its DataFrames immutable.
Inspection and `to_risk_model()` do not rerun the estimator.

`save_bundle` persists **inputs** to six CSVs in a new directory, refusing replacement.
`load_bundle` restores those inputs without fitting. There is no promised fitted
checkpoint serializer. Export individual result tables with pandas `to_csv` or
another caller-chosen format and retain units, date, recipe and source provenance
beside them. The example's `persist_example` writes inspectable tables,
`result_metadata.json` with conventions/provenance, and a PDF when its illustrative
audit is supported. Those output CSVs are not a fitted-object reload format.

## Separate downstream usage

`to_risk_model()` exports the selected total covariance to existing QIS risk
methods. It does not include all FL diagnostics or full residual history. With
correlated residuals, the authoritative total risk and the optional explicitly
labelled diagonal-reference decomposition have different meanings.

[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) receives supplied
alphas and covariance; [QIS](https://github.com/ArturSepp/QuantInvestStrats) receives
risk/portfolio/scenario inputs. TF's own GLS/CMA/covariance reports remain in TF and
use existing QIS plotting primitives. See the [handoff example](integration.md).
