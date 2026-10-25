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
| `FactorCovarEstimator.fit_current` | Factor histories, prepared log asset returns, FL settings, cutoff | `FactorModelSnapshot` |
| `FactorCovarEstimator.fit_rolling` | Same information plus rolling dates | `RollingFactorModel.snapshots`, keyed by cutoff |
| `compute_cma_diagnostics` | Supplied forecasts and fitted arrays | Separate `CmaDiagnosticResult` tables; does not revise the model |

Successful return means construction or fitting has completed its validation. An
invalid input or failed fit raises; the FM snapshot has no generic `status` field.
Do not invent an `optimal` status for a statistical fit. Inspect FL diagnostics
and residual histories to assess its statistical adequacy.

## Inspectable fields and units

| Field | Meaning and units |
|---|---|
| Caller configuration and `FactorCovarEstimator` | Recipe identity, currency/hedging preparation, cadences, transformations, loss scaling and residual policy |
| `risk_factor_prices` | Ordered NAVs or typed observed levels; LOG uses dimensionless returns, DIFFERENCE retains native points |
| `snapshot.loadings` | Independent labelled copy, assets × factors; decimal asset log return per factor return unit |
| `snapshot.factor_covariance` | Independent labelled copy of annual factor covariance in declared native coordinates |
| `snapshot.residual_covariance` | Selected annual asset residual covariance, including residual variance/correlation policy |
| `snapshot.asset_covariance` | Annual asset log-return covariance, preserving the full selected residual block |
| `snapshot.raw_diagonal_reference_covariance` | Annual orthogonal covariance with unit raw residual-variance weight; used for GLS reference risk |
| `snapshot.fit_statistics`, `snapshot.fit_diagnostics` | Raw fit statistics and optional span-free display table; no implicit historical-alpha span |
| `snapshot.residuals`, `snapshot.residual_metadata` | Native residual history and per-asset cadence, beta span, annualisation and stored/raw scale |
| `snapshot.residual_correlation`, `snapshot.residual_correlation_metadata` | Plain dimensionless correlation plus common residual returns, metadata, frequency/span and separate observation/estimation dates |
| `snapshot.clusters`, `.linkages`, `.cutoffs`, `.derived_signs` | Available fit diagnostics; per-frequency cluster/linkage/cutoff views are materialized too |
| `snapshot.cutoff`, `snapshot.provenance` | Information cutoff, latest observations, conventions and producer settings; retain caller-specific recipe, currency, units and source evidence separately |

Factor and asset labels and their ordering are substantive. Monthly covariance is
annualised with 12 observations per year; each supported cadence uses the existing
QIS/FL annualisation. Annual covariance contains variance units, not percent
volatility. Supplied asset returns must be log returns; total/excess basis must agree
with the factor basis. Input timestamps declare availability; the estimator
cannot discover provider publication delays or restatements.

The synthetic example produces two factors, four assets and a 4 × 4 annual asset
covariance at **2024-01-31**, using 121 monthly input level observations. It verifies
the independent identity $\Sigma = B F B^{\mathsf T} + R$ with explicit component sums.
The construction-only example retains its separate model specification; a loaded
CSV panel alone does not retain an executable reconstruction of its recipe.

## Ownership, mutability and persistence

In the **unreleased 0.3.0 candidate**, completed FM results own plain data and retain
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

Input loading and persistence belong to the caller. There is no package input
serializer or promised fitted checkpoint serializer. Export individual result tables with pandas `to_csv` or
another caller-chosen format and retain units, date, recipe and source provenance
beside them. The example's `persist_example` writes inspectable model tables and
`result_metadata.json` with conventions/provenance. Eligible models additionally
produce forecast tables, GLS/CMA results and a PDF. Other models retain their full
tables and an explicit unavailable-report reason. Those output CSVs are not a
fitted-object reload format.

## Interpretation and limitations

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
