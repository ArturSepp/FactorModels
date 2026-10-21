# factormodels

**FactorModels** package implements factor models for estimation of factor risk
exposures, factor-based covariance matrices, and stress testing.

The package defines the construction of factors including

1. **Multi-asset tradable factor models (MATF)**

   The construction uses net asset values (NAVs) of tradable futures and
   quantitative investment strategies (QIS), developed in

   - Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic
     and Tactical Asset Allocation for Multi-Asset Portfolios*,
     [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86)
   - Sepp, A., Hansen, E. and Kastenholz, M. (2027), *Capital Market Assumptions and
     Strategic Asset Allocation Using Multi-Asset Tradable Factors*,
     [SSRN 6785958](https://ssrn.com/abstract=6785958)

   Application: risk attribution and portfolio hedging using liquid instruments.

2. **Price- and level-based factors**

   Use market and economic variables such as the VIX, government yields and
   economic data.

   Application: risk attribution and scenario analysis using macro data.

Its standalone output is a completed `FactorModelSnapshot` with labelled loadings,
factor, residual and asset covariance, and provenance at a stated cutoff.
[Inspect a completed model](docs/completed_model.md), including units, dates,
residual policy and table export, with the
[canonical offline example](examples/reference_workflows.py). Statistical
estimation belongs to [factorlasso](https://github.com/ArturSepp/factorlasso), and
return preparation and risk export to [qis](https://github.com/ArturSepp/QuantInvestStrats).
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) consumes the
fitted covariance and is not required to complete a fit.

**Install (checkout):** `python -m pip install .` · **Import:** `factormodels` · **Status:** Release candidate

<!-- Enable at publication:
[![PyPI](https://img.shields.io/pypi/v/factormodels?style=flat-square)](https://pypi.org/project/factormodels/)
[![Python](https://img.shields.io/pypi/pyversions/factormodels?style=flat-square)](https://pypi.org/project/factormodels/)
[![Docs](https://readthedocs.org/projects/factormodels/badge/?version=latest)](https://factormodels.readthedocs.io/en/latest/)
[![Downloads](https://static.pepy.tech/badge/factormodels)](https://pepy.tech/project/factormodels)
-->
[![CI](https://github.com/ArturSepp/FactorModels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/FactorModels/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArturSepp/FactorModels.svg?style=flat-square)](LICENSE)

**Estimator:** loadings, sign constraints, priors and cluster penalties come from
[factorlasso](https://github.com/ArturSepp/factorlasso) (Sepp and Kastenholz 2026).
`factormodels` adds the financial model around it. See [Citation](#citation) for
the BibTeX entries.

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
the residual alpha. The same $\hat{B}$ enters the expected return and the risk model.
This is what makes the capital market assumptions coherent with the covariance used
to allocate against them.

`factormodels` is useful when four things matter:

- The factors are **your own definitions**. A basket of futures or indices held at a
  target volatility, a tracked index, or an observed level, each with a declared
  `LOG` or `DIFFERENCE` return type.
- The assets **report at different cadences**. Daily indices, monthly funds and
  quarterly private-market series are regressed on their own grid and assembled into
  one annual covariance.
- **No look-ahead.** Every fit states its cutoff and provenance, a supplied factor
  covariance declares when it became available, and the rolling fit is causal at
  every date.
- **Expected returns and risk share one loading matrix.** Factor premia are inferred
  from provider forecasts by GLS on the fitted loadings, and the audit accounts for
  the Sharpe ratio those forecasts imply under the fitted covariance.

It is written in pandas and numpy on top of `factorlasso` and `qis`. The regression
solver is CVXPY through `factorlasso` (default `CLARABEL`), so every loading is the
solution of an explicit, auditable problem.

## Installation

```bash
python -m pip install .
```

Requires Python ≥ 3.10, `factorlasso` ≥ 1.0.4.post1 (MIT), `qis` ≥ 5.38.1, and
numpy / pandas / scipy / matplotlib. Run the command above from a source checkout.
After the first index release, `pip install factormodels` installs from PyPI.

## Five-minute quickstart

Run the six Python examples below in order in one session; all inputs are synthetic.

```python
from factormodels import fit_bundle, synthetic_bundle

model = fit_bundle(synthetic_bundle())

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

`synthetic_bundle()` supplies 121 monthly USD NAV observations for two factors and
four assets, with a fixed seed. Returns are decimal log returns, covariances are
annualised with 12 periods a year, and the cutoff is the last observation. The
`FactorModelSnapshot` also carries `residual_covariance`, `provenance` and the
FactorLasso fit in `model.data`. `model.to_risk_model()` exports the selected
covariance as a `qis.RiskModel` without refitting. The loadings are LASSO
estimates over a 36-month span, so a small planted loading can shrink to zero.

### From the fitted model to factor premia

Expected returns are annual arithmetic excess returns and the risk model is built
from annualised log-return moments. The paper bridges the two with the second-order
rule $A\,(\bar r_{\log} + \tfrac{1}{2}\operatorname{Var}(r_{\log}))$, exported as
`annual_arithmetic_from_log`. The snippet assumes a 4% premium on each factor,
credits the historical residual alpha, tilts the asset forecasts, and recovers the
premia by GLS.

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
alpha_log = model.data.estimate_alpha(alpha_span=36)            # annualised log residual mean
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

`gls` is a `GlsPremiaResult` with premia, conditional Student-t intervals, the
asset-level reconciliation and the factor-mimicking portfolios. `audit` is a
`CmaDiagnosticResult` with the Sharpe accounting of the forecasts under the fitted
covariance. The [canonical example](examples/reference_workflows.py) records the
same assumptions in provenance and exports the tables and a report.

---

## Key differentiators

### 1. Factors as declared histories

A factor is a `Factor` with a return type. A basket or a tracked index uses `LOG`
returns of its NAV. An observed level uses `DIFFERENCE` changes in its native
units and is copied through construction unchanged.

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

The declared types travel with the fit. A supplied factor covariance in the wrong
coordinates is rejected, and the bundle schema records the units of every factor.

### 2. Baskets held at a volatility target

A basket factor is a long side, an optional short side, and the volatility the
combined basket is held at. The leverage $l_f(t) = \sigma^{*}_f / \hat\sigma_f(t)$
makes the regression loading read as an exposure ratio rather than as an artefact
of the volatility regime the factor was estimated in. A `FactorFamily` declares the
target once, so every regional equity factor is held at the same volatility and the
loadings are comparable across regions.

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

Construction runs through `qis.backtest_model_portfolio`, which holds units between
rebalancing dates. The result is a NAV, not a weighted average of returns.
Volatility targets are annualised decimals and the EWMA span counts weekly periods.
`compute_beta_loading_signs` turns the family rules into the sign-constraint matrix
that `factorlasso` consumes.

### 3. One annual covariance from assets at different cadences

Assets are supplied as a dictionary of return panels keyed by cadence. Each panel is
regressed on the factor returns sampled at its own grid, with its own regression span,
and the factor covariance is estimated once on the cadence you choose. The asset
covariance $B F B^{\top} + R$ is assembled in annual units.

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

The configuration is a frozen dataclass. Each call copies the `LassoModel` without
fitted state, so two fits never share a solver object.

### 4. Point-in-time fits with provenance

`fit_current(cutoff=...)` truncates every history before estimation, and
`fit_rolling(time_period=...)` produces one snapshot per rebalancing date, each
causal at its own date. A supplied factor covariance is a `DatedFactorCovariance`
with an observation date and an availability date, and it is rejected when either
lies after the cutoff. The test suite proves the contract by altering every
observation after the cutoff and asserting an identical fit.

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

### 5. Residual covariance policies retained in the result

The residual block $R$ is selected at fit time and travels with the snapshot. The
policies are `orthogonal` (diagonal), `empirical` (EWMA residual correlation on a
common grid, scaled by the current residual volatilities), and `exposure_cluster`
or `residual_cluster` (within-cluster mean correlation and zero between clusters,
with clusters taken from the fitted lasso groups or from Ward clustering of the
residuals). `residual_corr_weight` retains a fraction of the off-diagonal
correlation and `residual_var_weight` scales the block.
`snapshot.residual_covariance` returns the full selected $R$;
`snapshot.asset_covariance` returns the total $B F B^\top + R$.
`snapshot.to_risk_model()` exports it as a `qis.RiskModel`, and a diagonal reference
decomposition is produced only when `diagonal_reference=True` is requested.

### 6. GLS factor premia from provider forecasts

`infer_factor_premia(B, Omega, total_returns, cash_rate=..., alpha=..., alpha_weight=...)`
projects provider forecasts, net of cash and of the admitted alpha, onto the fitted
loadings by generalised least squares with the residual covariance as the weight
(Aitken 1935). The result carries the premia, conditional Student-t intervals, the
rank of the whitened design, the factor-mimicking portfolios and the asset-level
reconciliation. Forecast boxes and cash bounds give sensitivity ranges, and
`compute_gls_uncertainty`, `compute_gls_loading_stability` and
`compute_gls_provider_contrast` propagate declared forecast or loading uncertainty.

### 7. CMA audit and identification

`compute_cma_diagnostics` audits one frozen universe without changing its forecasts.
It reconciles each asset's forecast into loaded premia, admitted alpha and residual,
accounts for the implied squared Sharpe ratio under the fitted covariance, and
reports an explicit status for every check. `compute_cma_identification` reports
the precision of each premium and tests pairwise contrasts (common and spread)
against declared equivalence margins, conditional on the fixed model.
`compute_conditional_gls_intervals` gives pointwise intervals under the diagonal
working model with the residual scale estimated from the forecasts.
`reports.cma_diagnostics` renders the audit with `qis` plotting primitives.

### 8. A six-file bundle contract

`load_bundle`, `fit_bundle` and `save_bundle` are separate actions on a versioned
directory of six CSV files: factor levels, asset prices, FX spots, domestic rates,
asset metadata (currency, hedge ratio, return cadence) and settings. Currency
conversion and hedging run through `qis.FxRatesData`, the settings round-trip
through `BundleSettings`, and every fit records the SHA-256 of its inputs in
provenance. Loading never fits, fitting never writes, and saving refuses to
overwrite.

---

## Why factormodels

<!-- Mechanism paragraph. Draft for the author to rewrite in his own words. -->

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
Because the same estimated loading matrix enters the expected return
$\mu = \hat{B}\lambda + w \odot \alpha$ and the covariance $\hat{B} F \hat{B}^{\top} + R$,
alpha and beta are consistent by construction. This separates the approach from
building-block expected returns and from Black-Litterman posteriors, where the
return view and the risk model are estimated apart. `factormodels` is the
implementation that produces that loading matrix, the covariance around it, and the
premia implied by a set of forecasts.

---

## When to use it, and when not

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

---

## Examples

Runnable examples in [`examples/`](examples/), all offline:

- [`reference_workflows.py`](examples/reference_workflows.py): synthetic bundle,
  caller-supplied bundle and public-ETF proxy construction, with the fitted model
  connected to GLS premia and the CMA audit, and table export on request.
- [`construct_factors.py`](examples/construct_factors.py): factor definitions,
  NAV construction and loading-sign constraints from synthetic daily prices.
- [`gls_factor_premia.py`](examples/gls_factor_premia.py): a self-contained
  arithmetic calibration of the GLS inference with an independent QR reference.
- [`methods/factor_covariance_hcgl.py`](examples/methods/factor_covariance_hcgl.py):
  mixed-frequency estimation with explicit EWMA reference sums and strict-cutoff
  comparisons.
- [`downstream_handoff.py`](examples/downstream_handoff.py): fitted model to an
  `optimalportfolios` solve to `qis` tracking error. Requires `optimalportfolios`.

Method documentation: [completed model](docs/completed_model.md),
[factor construction](docs/construction.md),
[factor covariance](docs/factor_covariance_hcgl.md),
[GLS factor premia](docs/gls_factor_premia.md),
[CMA diagnostics](docs/cma_diagnostics.md),
[six-file bundles](docs/rolling_factor_covar_from_csv.md),
[downstream integration](docs/integration.md) and the [API reference](docs/api.md).

---

## Testing

```bash
uv sync --locked --group test --group lint
uv run --no-sync ruff check src tests tools
uv run --no-sync pytest
```

The suite covers construction identities against independent sums, point-in-time
contracts with deliberate counterexamples, residual-policy assembly, bundle
round-trips, GLS and audit identities against independent QR and Cholesky
references, and the documented examples.

---

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical full catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio optimisation with supplied alphas and covariance |
| [`factormodels`](https://github.com/ArturSepp/FactorModels) *(this package)* | Factor definitions, histories, fitted loadings and covariance |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation and numerical covariance containers |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain normalisation, reconstruction, querying, and visualisation |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Vectorised vanilla option pricers and implied volatility fitters |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic volatility pricing analytics |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Money-weighted multi-factor alpha from private-asset cash flows |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |

`factormodels` depends on `factorlasso` and `qis`. It does not import
`optimalportfolios`, and the import ban is enforced by `ruff` in CI.

## Feedback & contributing

- **Bug:** open an [issue](https://github.com/ArturSepp/FactorModels/issues) with the version, Python/platform, a minimal reproducer, and expected versus actual output.
- **Feature:** describe the modelling goal, the current workaround, and the smallest useful API. In particular: which construction rule, residual policy or diagnostic is missing?
- **Question or methodology:** open an issue and identify the method, paper section, or convention involved.
- **Contribution:** follow [CONTRIBUTING.md](CONTRIBUTING.md).

See [`CHANGELOG.md`](CHANGELOG.md) for release history.

---

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

If you use `factormodels` in academic work, please cite the papers that define the
model and its applications, the estimator it builds on, and the software itself:

```bibtex
@article{SeppHansenKastenholz2027MATF,
  author  = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title   = {Capital Market Assumptions and Strategic Asset Allocation Using
             Multi-Asset Tradable Factors},
  journal = {The Journal of Portfolio Management},
  year    = {2027},
  note    = {Forthcoming. SSRN 6785958, https://ssrn.com/abstract=6785958}
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

@article{SeppKastenholz2026factorlasso,
  author  = {Sepp, Artur and Kastenholz, Mika},
  title   = {{factorlasso}: Sparse Multi-Output Regression with
             Cluster-Grouped Sign Constraints in {Python}},
  journal = {Journal of Statistical Software},
  year    = {2026},
  note    = {Submitted.}
}

@software{factormodels,
  author  = {Sepp, Artur},
  title   = {factormodels: Multi-Asset Factor Models in {Python}},
  year    = {2026},
  version = {0.1.0rc4},
  url     = {https://github.com/ArturSepp/FactorModels},
}
```

---

## License

MIT, see [`LICENSE`](LICENSE). Dependencies and bundled libraries retain their own
licences, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the
[dependency licensing notes](docs/licensing.md). `factorlasso` is MIT from
`1.0.4.post1`, which is the declared floor.
