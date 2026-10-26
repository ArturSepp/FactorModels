# factormodels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

**Factor construction, GLS premia, capital market assumptions and factor covariance estimation
— multi-asset factor models in Python from tradable instruments and observed levels to
point-in-time loadings, covariance and provider-implied factor premia.**

**Start here:** [minimal model definition and rolling covariance](#minimal-model-definition-and-rolling-covariance).

Its standalone output is a completed `FactorModelSnapshot` with labelled loadings,
factor, residual and asset covariance, and provenance at a stated cutoff.
[Inspect a completed model](docs/completed_model.md), including units, dates,
residual policy and table export, with the
[canonical offline example](examples/reference_workflows.py). Statistical
estimation belongs to [factorlasso](https://github.com/ArturSepp/factorlasso), and
return preparation and risk export to [qis](https://github.com/ArturSepp/QuantInvestStrats).
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) consumes the
fitted covariance and is not required to complete a fit.

**Install:** `pip install factormodels` · **Import:** `factormodels` · **Status:** Released

[![PyPI](https://img.shields.io/pypi/v/factormodels?style=flat-square)](https://pypi.org/project/factormodels/)
[![Python](https://img.shields.io/pypi/pyversions/factormodels?style=flat-square)](https://pypi.org/project/factormodels/)
[![CI](https://github.com/ArturSepp/FactorModels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/FactorModels/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/factormodels/badge/?version=latest)](https://factormodels.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/FactorModels.svg?style=flat-square)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/factormodels)](https://pepy.tech/project/factormodels)
[![Monthly](https://static.pepy.tech/badge/factormodels/month)](https://pepy.tech/project/factormodels)

**Papers:** Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and
Tactical Asset Allocation for Multi-Asset Portfolios*, The Journal of Portfolio Management 52(4) ·
Sepp, A., Hansen, E. and Kastenholz, M. (2027), *Capital Market Assumptions and Strategic Asset
Allocation Using Multi-Asset Tradable Factors* · **Estimator:** Sepp, A. and Kastenholz, M.
(2026), *factorlasso*. See [Statement of need](#statement-of-need) for what each paper
contributes and [Citation](#citation) for the BibTeX entries.

**Documentation:** [Read the Docs](https://factormodels.readthedocs.io/en/latest/) · [completed model](docs/completed_model.md) ·
[reference workflows](docs/reference_workflows.md) · [API reference](docs/api.md)

---

## Why factormodels

Multi-asset allocation needs a factor model that serves two purposes at once. The
covariance has to price the risk of funds, indices and private-market series that
report at different frequencies, and the expected returns have to be stated in the
same coordinates so that an optimiser can trade one against the other. Commercial
risk systems (MSCI Barra, Axioma, Bloomberg's multi-asset-class models) give loadings
on vendor factors that the investor can neither hold nor forecast. Academic factor
returns are long-short portfolios without transaction costs or volatility scaling,
so their loadings change meaning with the volatility regime of the sample. The MATF
model defines each premium factor as a basket the investor can hold at a target
volatility. The loading is then an exposure ratio, the premium is a return the
investor can earn, and observed levels such as the VIX can be added for risk only.

The model is the linear factor decomposition

$$
r_i(t) = \alpha_i + \beta_i^{\top} f(t) + \varepsilon_i(t),
\qquad
\Sigma = B F B^{\top} + R,
\qquad
\mu^{excess} = B \lambda + w \odot \alpha,
$$

where $f(t)$ holds the factor returns in their declared coordinates (log returns of a
NAV, or differences of a level), $B$ is the asset-by-factor loading matrix estimated
at the information cutoff, $F$ is the annual factor covariance, $R$ is the selected
residual covariance, $\lambda$ are the factor premia and $w$ the admission weights on
the residual alpha. Because the same estimated loading matrix enters the expected return
and the covariance, alpha and beta are consistent by construction. This separates the
approach from building-block expected returns and from Black-Litterman posteriors, where
the return view and the risk model are estimated apart. `factormodels` is the
implementation that produces that loading matrix, the covariance around it, and the
premia implied by a set of forecasts. It is written in pandas and numpy on top of
`factorlasso` and `qis`; the regression solver is CVXPY through `factorlasso` (default
`CLARABEL`), so every loading is the solution of an explicit, auditable problem.

### Key differentiators

**1. Factors as declared histories.** A `Factor` carries a return type: a basket or a tracked
index uses `LOG` returns of its NAV, an observed level uses `DIFFERENCE` changes in its native
units and is copied through construction unchanged. The declared types travel with the fit; a
supplied factor covariance in the wrong coordinates is rejected. Callers retain the native
units of every factor alongside their inputs. See [factor construction](docs/construction.md).

**2. Baskets held at a volatility target.** A basket factor is a long side, an optional short
side, and the volatility the combined basket is held at. The leverage
$l_f(t) = \sigma^{*}_f / \hat\sigma_f(t)$ makes the regression loading read as an exposure ratio
rather than an artefact of the volatility regime the factor was estimated in; a `FactorFamily`
declares the target once, so loadings are comparable across regions. Construction runs through
`qis.backtest_model_portfolio`, which holds units between rebalancing dates, and
`compute_beta_loading_signs` turns the family rules into the sign-constraint matrix
`factorlasso` consumes.

**3. One annual covariance from assets at different cadences.** Assets are supplied as a
dictionary of return panels keyed by cadence. Each panel is regressed on the factor returns
sampled at its own grid, with its own regression span, and the factor covariance is estimated
once on the cadence you choose; $B F B^{\top} + R$ is assembled in annual units. The
`FactorCovarEstimator` configuration is a frozen dataclass and every fit copies the
`LassoModel` without fitted state. See [factor covariance](docs/factor_covariance_hcgl.md).

**4. Point-in-time fits with provenance.** `fit_current(cutoff=...)` truncates every history
before estimation and `fit_rolling(time_period=...)` produces one snapshot per rebalancing date,
each causal at its own date. A supplied `DatedFactorCovariance` carries an observation date and
an availability date and is rejected when either lies after the cutoff. The test suite proves the
contract by altering every observation after the cutoff and asserting an identical fit.

**5. Residual covariance policies retained in the result.** The residual block $R$ is selected
at fit time and travels with the snapshot: `orthogonal` (diagonal), `empirical` (EWMA residual
correlation on a common grid, scaled by current residual volatilities), `exposure_cluster` or
`residual_cluster` (within-cluster mean correlation, zero between clusters).
`residual_corr_weight` and `residual_var_weight` scale the off-diagonal and the whole block;
`snapshot.to_risk_model()` exports the selected covariance as a `qis.RiskModel`.

**6. GLS factor premia from provider forecasts.** `infer_factor_premia` projects provider
forecasts, net of cash and of the admitted alpha, onto the fitted loadings by generalised least
squares with the residual covariance as the weight (Aitken 1935), returning premia, conditional
Student-t intervals, the rank of the whitened design, the factor-mimicking portfolios and the
asset-level reconciliation; `compute_gls_uncertainty`, `compute_gls_loading_stability` and
`compute_gls_provider_contrast` propagate declared uncertainty. See
[GLS factor premia](docs/gls_factor_premia.md).

**7. CMA audit and identification.** `compute_cma_diagnostics` reconciles each asset's forecast
into loaded premia, admitted alpha and residual, accounts for the implied squared Sharpe ratio
under the fitted covariance, and reports an explicit status for every check;
`compute_cma_identification` tests pairwise premium contrasts against declared equivalence
margins; `compute_conditional_gls_intervals` gives pointwise intervals under the diagonal
working model. `reports.cma_diagnostics` renders the audit with `qis` plotting primitives. See
[CMA diagnostics](docs/cma_diagnostics.md).

### When to use it — and when not

**Use it when:**

- Your universe mixes asset classes, vehicles and reporting cadences, and you need
  one annual covariance with stated units and a stated cutoff.
- You define the factors yourself, as baskets, tracked indices or observed levels,
  and you want the loadings to be hedge ratios rather than regression artefacts.
- You need expected returns and risk from one loading matrix, with factor premia
  inferred from provider forecasts and audited for the Sharpe ratio they imply.
- You run the estimation on a schedule and need every snapshot to be reproducible
  from recorded inputs and provenance.

**Reach for something else when:**

- You want the sparse multi-output estimator on its own. Use `factorlasso`.
- You want a characteristic-based model for thousands of single stocks. The
  cross-sectional regression of a Barra-type model is a different estimator.
- You need portfolio optimisation. Pass the fitted covariance to `optimalportfolios`.
- You need non-linear or regime-switching factor dynamics.

The current release is **0.3.0**. Completed-result callers
should follow the [migration guide](docs/migration.md).

## Installation

```bash
python -m pip install factormodels
```

Requires Python ≥ 3.11, `factorlasso` ≥ 1.0.4.post1 (MIT), `qis` ≥ 5.38.1, and
numpy / pandas / scipy / matplotlib. For a source checkout, use `python -m pip install .`.
Dependency licences are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the
[dependency licensing notes](docs/licensing.md).

## Table of contents

1. [Why factormodels](#why-factormodels)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Citation](#citation)
11. [Ecosystem](#ecosystem)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Statement of need

A factor model for multi-asset allocation must be estimable on short, mixed-frequency histories
of funds, indices and private-market series, must be causal at every date it is used on, and
must state its expected returns in the coordinates of its covariance. `factormodels` implements
the factor construction, estimation and premia inference of the papers below on top of the
`factorlasso` estimator; the two kinds of factor it defines are **multi-asset tradable factors
(MATF)** built from net asset values of tradable futures and quantitative investment strategies,
for risk attribution and portfolio hedging using liquid instruments, and **price- and
level-based factors** such as the VIX, government yields and economic data, for risk attribution
and scenario analysis using macro data.

| Paper | Implemented analytics | Where |
|---|---|---|
| Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86) | Tradable-factor construction as volatility-targeted baskets, sign-constrained loadings and the factor covariance that feed the ROSAA risk model | [factor construction](docs/construction.md), [factor covariance](docs/factor_covariance_hcgl.md) |
| Sepp, A., Hansen, E. and Kastenholz, M. (2027), *Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors*, [SSRN 6785958](https://ssrn.com/abstract=6785958) | Factor premia inferred from provider forecasts by GLS on the fitted loadings, the arithmetic/log bridge `annual_arithmetic_from_log`, and the CMA audit of the Sharpe ratio the forecasts imply under the fitted covariance | [GLS factor premia](docs/gls_factor_premia.md), [CMA diagnostics](docs/cma_diagnostics.md) |
| Sepp, A. and Kastenholz, M. (2026), *factorlasso: Sparse Multi-Output Regression with Cluster-Grouped Sign Constraints in Python*, submitted to the Journal of Statistical Software | The estimator behind every loading: sign constraints, priors and cluster penalties | [`factorlasso`](https://github.com/ArturSepp/factorlasso) |

## Overview

`factormodels` adds the financial model around the `factorlasso` estimator: factor definitions
and NAV construction, the mixed-cadence `FactorCovarEstimator`, the completed
`FactorModelSnapshot`, GLS premia and CMA diagnostics, and the
export to a `qis.RiskModel`. The [API reference](docs/api.md) lists every public object and the
[downstream integration](docs/integration.md) page shows the handoff to `optimalportfolios`.

### Methods

| Method | Formula | Implementation | Article · example |
|---|---|---|---|
| Volatility-targeted basket factors | $l_f(t) = \sigma^{*}_f / \hat\sigma_f(t)$, NAV through `qis.backtest_model_portfolio` | `Factor`, `FactorFamily`, `VolTargetPolicy`, `assemble_factor_model`, `build_factor_panel`, `compute_beta_loading_signs` | [construction](docs/construction.md) · [`construct_factors.py`](examples/construct_factors.py) |
| Point-in-time loadings and covariance | $\Sigma = B F B^{\top} + R$, annual units, per-cadence regression spans | `FactorCovarEstimator.fit_current`, `.fit_rolling`, `DatedFactorCovariance`, `FactorModelSnapshot` | [factor covariance](docs/factor_covariance_hcgl.md) · [`methods/factor_covariance_hcgl.py`](examples/methods/factor_covariance_hcgl.py) |
| Residual covariance policies | $R$ orthogonal, empirical, exposure-cluster or residual-cluster; `residual_corr_weight`, `residual_var_weight` | `FactorCovarEstimator(residual_type=...)`, `snapshot.residual_covariance`, `snapshot.to_risk_model()` | [completed model](docs/completed_model.md) · [`reference_workflows.py`](examples/reference_workflows.py) |
| GLS factor premia | $\mu^{excess} = B \lambda + w \odot \alpha$ solved by GLS with $R$ as weight; bridge $A\,(\bar r_{\log} + \tfrac{1}{2}\operatorname{Var}(r_{\log}))$ | `infer_factor_premia` → `GlsPremiaResult`, `annual_arithmetic_from_log`, `compute_gls_uncertainty`, `compute_gls_loading_stability`, `compute_gls_provider_contrast` | [GLS factor premia](docs/gls_factor_premia.md) · [`gls_cma_workflow.py`](examples/gls_cma_workflow.py), [`methods/gls_factor_premia.py`](examples/methods/gls_factor_premia.py) |
| CMA audit and identification | forecast = loaded premia + admitted alpha + residual; implied squared Sharpe under $\Sigma$ | `compute_cma_diagnostics` → `CmaDiagnosticResult`, `compute_cma_identification`, `compute_conditional_gls_intervals`, `reports.cma_diagnostics` | [CMA diagnostics](docs/cma_diagnostics.md) · [`reference_workflows.py`](examples/reference_workflows.py) |

### Analytics at a glance

| Area | What the package computes |
|---|---|
| Factor construction | Long/short baskets held at a family volatility target, tracked indices, observed levels with `LOG` or `DIFFERENCE` return types; sign-constraint matrices from family rules |
| Loadings and covariance | Per-cadence sign-constrained LASSO loadings, annual factor covariance on a chosen grid, asset covariance $B F B^{\top} + R$, rolling causal snapshots with provenance (cutoff, observation dates, spans) |
| Residual risk | Orthogonal, empirical, exposure-cluster and residual-cluster residual covariances; diagonal reference decomposition on request |
| Expected returns | GLS factor premia with Student-t intervals, factor-mimicking portfolios, asset-level reconciliation, forecast-box and cash-bound sensitivity ranges, propagated forecast and loading uncertainty |
| Audit | CMA reconciliation and Sharpe accounting with explicit per-check status; premium precision and pairwise contrasts against equivalence margins |
| Data contract | Supplied factor histories and log asset-return DataFrames by cadence; export of the selected covariance as `qis.RiskModel` |

## Five-minute quickstart

### Minimal model: definition and rolling covariance

Start with [`minimal_model.py`](examples/minimal_model.py): define two factors,
construct their NAVs, estimate three assets at four quarterly cutoffs, and retrieve
an asset covariance by date. The self-contained code below uses public APIs available
in `factormodels` 0.1.0. Inputs are synthetic monthly USD log returns and total-return
NAVs; each factor holds one source without volatility targeting or cash/FX adjustment.

```python
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType

from factormodels import (
    Factor, FactorCovarEstimator, FactorFamily, FactorModelSpec,
    RiskFactors, VolTargetPolicy, build_factor_panel,
)


class MinimalFactors(RiskFactors):
    """Ordered factor identifiers for this two-factor model."""

    EQUITY = "Equity"
    RATES = "Rates"

# 1. Define each factor as a single-source basket, without volatility targeting.
spec = FactorModelSpec(
    name="Minimal two-factor model", factors=MinimalFactors, file_name="minimal_model",
    baskets={
        member: Factor(
            name=member.value, family=FactorFamily(member.value, vol_target=None),
            long_side_tickers=(f"{member.value} source",), long_side_weights=(1.,),
            vol_target_policy=VolTargetPolicy.NONE,
        )
        for member in MinimalFactors
    },
)

# Synthetic monthly inputs; replace these two panels with your own histories.
dates = pd.date_range("2015-12-31", "2024-12-31", freq="ME")
rng = np.random.default_rng(42)
factor_log_returns = rng.normal([.005, .002], [.04, .02], (len(dates) - 1, 2))
source_prices = pd.DataFrame(
    100. * np.exp(np.vstack([np.zeros(2), factor_log_returns.cumsum(axis=0)])),
    index=dates, columns=["Equity source", "Rates source"],
)
# These generating loadings are only for simulation; the estimator fits its own.
loadings = np.array([[.9, .1], [.1, .8], [.5, .4]])
asset_returns = pd.DataFrame(
    factor_log_returns @ loadings.T + rng.normal(0., .01, (len(dates) - 1, 3)),
    index=dates[1:], columns=["Equity asset", "Bond asset", "Balanced asset"],
)
factor_prices = build_factor_panel(
    spec, source_prices, returns_freq="ME", rebalancing_freq="ME",
)

# 2. Estimate loadings and covariance at each quarter-end, using monthly inputs.
estimator = FactorCovarEstimator(
    lasso_model=LassoModel(
        model_type=LassoModelType.LASSO, span=36, warmup_period=6,
        reg_lambda=1e-5, solver="CLARABEL",
    ),
    factor_returns_freq="ME", factor_covar_span=36,
    rebalancing_freq="QE", residual_type="orthogonal",
)
rolling = estimator.fit_rolling(
    risk_factor_prices=factor_prices, asset_returns_dict={"ME": asset_returns},
    time_period=qis.TimePeriod("2024-03-31", "2024-12-31"),
    assets=list(asset_returns.columns),
)

# 3. Retrieve the full asset covariance by an exact fitted date, or the latest date.
covariances = rolling.covariances  # dict[pd.Timestamp, pd.DataFrame]
as_of = pd.Timestamp("2024-06-30")
covariance = covariances[as_of]
latest_date = max(covariances)
latest_covariance = covariances[latest_date]
print(spec.factor_names())
print([str(date.date()) for date in covariances])
print(f"Selected {as_of.date()}: {covariance.shape}")
print(f"Latest {latest_date.date()}: {latest_covariance.shape}")
```

```text
['Equity', 'Rates']
['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']
Selected 2024-06-30: (3, 3)
Latest 2024-12-31: (3, 3)
```

Each retrieved DataFrame is the full annualised asset log-return covariance in the
supplied asset order, including factor risk and the selected diagonal residual block.
The 36-observation EWMA spans weight expanding histories through each cutoff;
they are not fixed 36-month windows. Retrieval uses an exact fitted date.

From a source checkout, run the full example to print the covariance matrices:

```bash
python -m examples.minimal_model
```

<a id="fit-a-supplied-bundle"></a>

### Fit supplied DataFrames directly

All inputs are synthetic. The local `synthetic_prices()` function below supplies two DataFrames with 121 monthly USD NAV observations for two
factors and four assets, with a fixed seed. Prepare returns and configure the estimator explicitly.
The repository workflow in `examples/reference_workflows.py` uses the same input generator.

```python
import qis
from factorlasso import LassoModel, LassoModelType
from factormodels import FactorCovarEstimator


def synthetic_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent synthetic factor NAVs and asset total-return prices.

    The fixed seed produces 121 monthly USD observations from January 2014 through
    January 2024 for two factors and four assets. These are illustrative prices,
    not market data. Callers prepare log returns and configure their estimator.
    No acquisition, fitting, rendering or persistence occurs.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Factor NAVs, then asset prices.
    """
    rng = np.random.default_rng(6102026)
    dates = pd.date_range("2014-01-31", periods=121, freq="ME")
    x = rng.normal([.004, .001], [.035, .018], (120, 2))
    b = np.array([[.9, .1], [.1, .9], [.6, .4], [.4, -.2]])
    y = x @ b.T + rng.normal(0., .013, (120, 4))
    levels = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(2), x.cumsum(axis=0)])),
                          index=dates, columns=["Equity", "Rates"])
    prices = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(4), y.cumsum(axis=0)])),
                          index=dates, columns=["Asset A", "Asset B", "Asset C", "Asset D"])
    return levels, prices

factor_navs, asset_prices = synthetic_prices()
asset_returns = qis.to_returns(asset_prices, freq="ME", is_log_returns=True, ffill_nans=False)
current_estimator = FactorCovarEstimator(
    lasso_model=LassoModel(
        model_type=LassoModelType.LASSO, span=36, warmup_period=6,
        reg_lambda=1e-5, demean=True, solver="CLARABEL", loss_normalization="sample",
    ),
    factor_returns_freq="ME", factor_covar_span=36, residual_type="orthogonal",
)
model = current_estimator.fit_current(
    risk_factor_prices=factor_navs, asset_returns_dict={"ME": asset_returns},
    assets=list(asset_returns.columns), cutoff=factor_navs.index[-1],
)

print(model.cutoff.date())
print(model.loadings.round(3))
print(model.factor_covariance.round(6))
print(model.asset_covariance.round(6))
```

```text
2024-01-31
         Equity  Rates
Asset A   0.926  0.169
Asset B   0.082  0.829
Asset C   0.526  0.390
Asset D   0.343 -0.000
          Equity     Rates
Equity  0.014735  0.000077
Rates   0.000077  0.004587
          Asset A   Asset B   Asset C   Asset D
Asset A  0.014411  0.001822  0.007512  0.004682
Asset B  0.001822  0.005197  0.002156  0.000435
Asset C  0.007512  0.002156  0.006363  0.002665
Asset D  0.004682  0.000435  0.002665  0.003494
```

Returns are decimal log returns, covariances are annualised with 12 periods a year, and the
cutoff is the last observation. The `FactorModelSnapshot` also carries `residual_covariance`,
`provenance` and FM-owned fit statistics, residual histories and cluster diagnostics. `model.to_risk_model()` exports the
selected covariance as a `qis.RiskModel` without refitting. The loadings are LASSO estimates
over a 36-month span, so a small planted loading can shrink to zero.

## Examples

### Highlighted examples

**From the fitted model to factor premia.** Expected returns are annual arithmetic excess
returns and the risk model is built from annualised log-return moments. The paper bridges the
two with the second-order rule $A\,(\bar r_{\log} + \tfrac{1}{2}\operatorname{Var}(r_{\log}))$,
exported as `annual_arithmetic_from_log`. The snippet continues from the quickstart model,
assumes a 4% premium on each factor, credits the historical residual alpha, tilts the asset
forecasts, and recovers the premia by GLS.

```python
import numpy as np
import pandas as pd
from factormodels import (
    CmaDiagnosticInputs, annual_arithmetic_from_log, compute_cma_diagnostics,
    infer_factor_premia,
)

B, F = model.loadings, model.factor_covariance
R, S = model.residual_covariance, model.asset_covariance

premia_assumed = pd.Series(0.04, index=B.columns)              # annual arithmetic
from factormodels import estimate_historical_alpha

alpha_log = estimate_historical_alpha(model, alpha_span=36)            # annualised log residual mean
asset_log_mean = B @ (premia_assumed - 0.5 * pd.Series(np.diag(F), index=B.columns)) + alpha_log
asset_mean = annual_arithmetic_from_log(asset_log_mean, pd.Series(np.diag(S), index=B.index))
alpha = asset_mean - B @ premia_assumed
forecast = asset_mean + pd.Series(np.linspace(-0.0025, 0.0025, len(B)), index=B.index)

gls = infer_factor_premia(B, R, forecast, cash_rate=0.0, alpha=alpha, alpha_weight=1.0)
audit = compute_cma_diagnostics(CmaDiagnosticInputs(
    betas=B, residual_variances=pd.Series(np.diag(R), index=B.index),
    factor_covariance=F, production_covariance=S,
    excess_cma=forecast, factor_premia=gls.premia, admitted_alpha=alpha,
    channels=pd.DataFrame({"admitted_alpha": alpha, "forecast_residual": gls.assets.residual}),
    provenance={"risk_basis": "annualised_log_return_covariance"}))

print(gls.factors[["estimate", "lower", "upper"]].round(4))
print(audit.summary["accounting_status"])
```

```text
        estimate   lower   upper
Equity    0.0393  0.0288  0.0497
Rates     0.0395  0.0261  0.0530
pass
```

`gls` is a `GlsPremiaResult` with premia, conditional Student-t intervals, the asset-level
reconciliation and the factor-mimicking portfolios. `audit` is a `CmaDiagnosticResult` with the
Sharpe accounting of the forecasts under the fitted covariance. The
[canonical example](examples/reference_workflows.py) records the same assumptions in provenance
and exports the tables and a report.

**From construction to a dated covariance.** The four blocks below continue in the same session
and are executed in order by `tests/test_readme.py`, which also asserts the strict-cutoff
contract on their output. Factors are declared histories with a return type; a basket is held at
its family's volatility target; assets at different cadences are regressed on their own grid and
assembled into one annual covariance; and a supplied factor covariance carries observation and
availability dates that the fit checks against its cutoff.

```python
from qis import ReturnTypes
from factormodels import Factor, FactorFamily, VolTargetPolicy

equity = Factor(name="Equity", family=FactorFamily("Equity", vol_target=0.10, long_only_sign=1.),
                long_side_tickers=("ES", "NQ"), long_side_weights=(0.5, 0.5))

vix = Factor(name="VIX", family=FactorFamily("Volatility", vol_target=None),
             long_side_tickers=("VIX",), long_side_weights=(1.,),
             vol_target_policy=VolTargetPolicy.NONE, is_observed_index=True,
             return_type=ReturnTypes.DIFFERENCE)
```

```python
from factormodels import RiskFactors, assemble_factor_model, build_factor_panel

# Deterministic teaching prices; these are not market observations.
dates = pd.bdate_range("2010-01-01", "2022-09-30")
rng = np.random.default_rng(42)
prices = pd.DataFrame(
    100.0 * np.exp(np.cumsum(rng.normal(0.0001, [0.010, 0.012, 0.005], (len(dates), 3)), axis=0)),
    index=dates, columns=["ES", "NQ", "TY"],
)

class Factors(RiskFactors):
    EQUITY = "Equity"
    RATES = "Rates"

rates = Factor(name="Rates", family=FactorFamily("Rates", vol_target=0.05),
               long_side_tickers=("TY",), long_side_weights=(1.,))
spec = assemble_factor_model(name="Teaching model", factors=Factors, file_name="teaching",
                             catalogue={"Equity": equity, "Rates": rates})
panel = build_factor_panel(spec, prices, rebalancing_freq="QE", returns_freq="W-WED")
```

Construction runs through `qis.backtest_model_portfolio`, which holds units between rebalancing
dates; the result is a NAV, not a weighted average of returns. Volatility targets are annualised
decimals and the EWMA span counts weekly periods.

```python
from factorlasso import LassoModel, LassoModelType
from factormodels import FactorCovarEstimator

import qis

factor_prices = panel
# Separate synthetic assets at monthly and quarterly reporting cadences.
monthly_returns = qis.to_returns(
    prices[["ES", "NQ"]].rename(columns={"ES": "Asset A", "NQ": "Asset B"}),
    freq="ME", is_log_returns=True, drop_first=True,
)
quarterly_returns = qis.to_returns(
    prices[["TY"]].rename(columns={"TY": "Asset C"}),
    freq="QE", is_log_returns=True, drop_first=True,
)
returns_by_cadence = {"ME": monthly_returns, "QE": quarterly_returns}

estimator = FactorCovarEstimator(
    lasso_model=LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                           span_freq_dict={"ME": 36, "QE": 12}, warmup_period=6,
                           solver="CLARABEL"),
    factor_returns_freq="W-WED",   # factor covariance grid
    factor_covar_span=52,          # EWMA span in weeks
    residual_type="orthogonal",
)
# factor_prices: daily factor NAVs such as `panel` above; the asset panels hold log returns
snapshot = estimator.fit_current(factor_prices, {"ME": monthly_returns, "QE": quarterly_returns},
                                 cutoff="2021-12-31")
```

```python
import qis
from factormodels import DatedFactorCovariance

rolling = estimator.fit_rolling(factor_prices, returns_by_cadence,
                                time_period=qis.TimePeriod("2018-12-31", "2021-12-31"),
                                rebalancing_freq="QE")
rolling.snapshots[pd.Timestamp("2021-12-31")].provenance   # cutoff, observation dates, spans

# Estimate the supplied matrix only from information available on its observation date.
matrix = estimator.fit_current(
    factor_prices, returns_by_cadence, cutoff="2021-11-30",
).factor_covariance

supplied = DatedFactorCovariance(matrix, observation_date="2021-11-30", available_date="2021-12-15")
snapshot = estimator.fit_current(factor_prices, returns_by_cadence, cutoff="2021-12-31",
                                 factor_covariance=supplied)
```

`fit_current(cutoff=...)` truncates every history before estimation and `fit_rolling` produces
one snapshot per rebalancing date, each causal at its own date; a supplied covariance whose
observation or availability date lies after the cutoff is rejected.

### Runnable examples

Runnable examples in [`examples/`](examples/), all offline:

- [`minimal_model.py`](examples/minimal_model.py): start here for factor definition,
  rolling estimation and covariance retrieval by date, using synthetic monthly inputs.
- [`reference_workflows.py`](examples/reference_workflows.py): synthetic DataFrames,
  caller-supplied price CSVs and public-ETF proxy construction, with the fitted model
  connected to GLS premia and the CMA audit, and table export on request.
- [`construct_factors.py`](examples/construct_factors.py): factor definitions,
  NAV construction and loading-sign constraints from synthetic daily prices.
- [`gls_cma_workflow.py`](examples/gls_cma_workflow.py): a self-contained
  arithmetic calibration of the GLS inference with an independent QR reference.
- [`methods/factor_covariance_hcgl.py`](examples/methods/factor_covariance_hcgl.py):
  mixed-frequency estimation with explicit EWMA reference sums and strict-cutoff
  comparisons.
- [`methods/gls_factor_premia.py`](examples/methods/gls_factor_premia.py): offline synthetic
  prices → LASSO loadings and alpha → provider-implied GLS premia, teaching the two stages on
  monthly excess-return NAVs.
- [`downstream_handoff.py`](examples/downstream_handoff.py): fitted model to an
  `optimalportfolios` solve to `qis` tracking error. Requires OP >=9.0.1 alongside FM >=0.3.0; see the [installation and handoff guide](docs/integration.md).

Method documentation: [completed model](docs/completed_model.md),
[factor construction](docs/construction.md),
[factor covariance](docs/factor_covariance_hcgl.md),
[GLS factor premia](docs/gls_factor_premia.md),
[CMA diagnostics](docs/cma_diagnostics.md),
[estimation from CSVs](docs/rolling_factor_covar_from_csv.md),
[downstream integration](docs/integration.md) and the [API reference](docs/api.md).

## Testing

First follow the external-checkout and environment setup in
[CONTRIBUTING.md](CONTRIBUTING.md). The commands below assume that setup, including
`UV_PROJECT_ENVIRONMENT`; do not run them with uv's default environment inside a
OneDrive checkout.

```bash
uv sync --locked --group test --group lint
uv run --no-sync pytest
uv run --no-sync ruff check src tests tools
```

The suite covers construction identities against independent sums, point-in-time
contracts with deliberate counterexamples, residual-policy assembly, direct
DataFrame and CSV workflows, GLS and audit identities against independent QR and Cholesky
references, and the documented examples. The `optimalportfolios` import ban is enforced by
`ruff` in CI.

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** use the [bug-report form](https://github.com/ArturSepp/FactorModels/issues/new?template=bug_report.yml) with the version, Python/platform, a minimal reproducer, and expected versus actual output.
- **Feature:** use the [feature-request form](https://github.com/ArturSepp/FactorModels/issues/new?template=feature_request.yml) and describe the modelling goal, the current workaround, and the smallest useful API. In particular: which construction rule, residual policy or diagnostic is missing?
- **Question or methodology:** open an [issue](https://github.com/ArturSepp/FactorModels/issues) and identify the method, paper section, or convention involved.
- **Contribution:** follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

A machine-readable citation is available in [`CITATION.cff`](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

If you use `factormodels` in academic work, please cite the software, the papers that define
the model and its applications, and the estimator it builds on:

```bibtex
@software{factormodels,
  author  = {Sepp, Artur},
  title   = {factormodels: Multi-Asset Factor Models in {Python}},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/ArturSepp/FactorModels},
}

@article{SeppOssaKastenholz2026,
  author  = {Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title   = {Robust Optimization of Strategic and Tactical Asset Allocation
             for Multi-Asset Portfolios},
  journal = {The Journal of Portfolio Management},
  year    = {2026},
  volume  = {52},
  number  = {4},
  pages   = {86--120},
}

@article{SeppHansenKastenholz2027MATF,
  author  = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title   = {Capital Market Assumptions and Strategic Asset Allocation Using
             Multi-Asset Tradable Factors},
  journal = {The Journal of Portfolio Management},
  year    = {2027},
  note    = {Forthcoming. SSRN 6785958, https://ssrn.com/abstract=6785958}
}

@article{SeppKastenholz2026factorlasso,
  author  = {Sepp, Artur and Kastenholz, Mika},
  title   = {{factorlasso}: Sparse Multi-Output Regression with
             Cluster-Grouped Sign Constraints in {Python}},
  journal = {Journal of Statistical Software},
  year    = {2026},
  note    = {Submitted.}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`factormodels`](https://github.com/ArturSepp/FactorModels) *(this package)* | Factor definitions, histories, fitted loadings and covariance |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

`factormodels` depends on `factorlasso` and `qis`. It does not import
`optimalportfolios`, and the import ban is enforced by `ruff` in CI.

## License

MIT, see [`LICENSE`](LICENSE). Dependencies and bundled libraries retain their own
licences, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the
[dependency licensing notes](docs/licensing.md). `factorlasso` is MIT from
`1.0.4.post1`, which is the declared floor.

## Disclaimer

`factormodels` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE](LICENSE) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
