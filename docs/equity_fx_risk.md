# Equity FX payoff risk

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Equity payoff risk combines translated local arithmetic excess payoffs with an
explicit currency-forward block. The research API retains every joint covariance
cross block and separates historical reconstruction from conditional forecasting.

## Overview

Fit translated SIMPLE payoffs on arithmetic market and style baskets, then add the
funded currency exposure using the same opening cash and hedge conventions. QIS
supplies return conversion, moment kernels and portfolio risk analytics. FactorLasso
fits loadings through FactorModels. Provider integration and book-specific basket
construction remain downstream responsibilities.

This is a research implementation. Exact payoff algebra and numerical covariance
reconciliation do not establish forecast accuracy or observed-equity qualification.
There is no production equity activation or accounting engine in this interface.

## Inputs, notation, and assumptions

For an interval, rL is the local asset total return, cL and cR are local/reference
cash returns, xFX is the spot appreciation of the local currency, and h is the
additional opening-principal hedge. Currency means quote currency, not domicile.
Rates and hedges are opening-known. Embedded share-class hedges remain in prices.
The forward is synthetic covered interest parity (CIP), with matching cash legs.

The translated response p is (rL-cL)(1+xFX), per opening reference notional. Factors
x are arithmetic baskets of these responses with declared opening reference-notional
weights. The market basket sums to one; signed style baskets declare net/gross
normalisation, neutrality and reset rules. Cumulative DIFFERENCE levels are only
an adapter to the fitted-model interface, not wealth, funded NAVs or logged payoffs.

All histories have one complete declared grid. Every moment uses the same trailing
complete-case mask, centering, covariance kernel and annualisation. Missing rows are
not forward-filled for estimation. Currency metadata, publication availability and
hedge instructions must be available at the declared cutoff. Finite positive cash
gross growth is required; local cash inputs are interval returns, not annual quotes.

## Methodology

### Exact payoff and covariance maps

Write the arithmetic forward payoff as phi = xFX + 1 - (1+cR)/(1+cL). For each
stock, the funded reference SIMPLE excess satisfies:

$$
e^R = p + (1+c_L-h)\varphi.
$$

With quote-currency incidence E, the opening exposure is H = E diag(1+cL-h).
Reference-currency stocks have no direct FX column. Full principal hedging h=1
leaves cL in this coefficient; hedging the local debt maturity h=1+cL removes it.

After a fit p=Bx+epsilon, define z=[x,epsilon,phi] and A=[B,I,H]. The conditional
covariance is:

$$
\Sigma^R = A\operatorname{Cov}(z)A^\top.
$$

Factor/residual, factor/FX and residual/FX covariance are all retained. Shrinkage,
constraints or mismatched regression/moment weights do not justify setting the
factor/residual block to zero. The implementation recomputes residuals with the
completed B, estimates full common-sample moments and explicitly replaces the
original fit's selected marginals. It validates joint positive semidefiniteness
(PSD), with roundoff tolerance, without clipping or scaling any cross block.

### Forecast and historical exposures

`covariance` fixes H at the cutoff for a declared next valuation date. Its cash
horizon and current metadata must be known at that opening. A future rolling
forecast must reconstruct H at each decision date.

`historical_covariance` uses each observation's dated opening H. The joint history
also carries the actual currency contribution g_t=H_t phi_t; the projection
[B,I,0,I] on [x,epsilon,phi,g] yields the historical SIMPLE covariance. This remains
exact when cash rates, interval lengths, currency metadata or hedges change.
It is a descriptive reconstruction, not a forecast using future H. With every cross block retained and the same moment kernel, this assembly reproduces
direct return covariance; the factor fit alone does not add covariance regularisation
or establish a forecasting improvement.

The fixed-H forecast is compared separately with direct fixed-exposure returns.
Its difference from actual dated USD-history covariance is an exposure-policy
change, not an algebraic approximation error. `without_residual_fx` is a labelled
fixed-H omission diagnostic; QIS export rejects it if its asset covariance is not PSD.

## Worked example

For local total return 10%, local cash 5%, reference cash 2%, spot appreciation
20% and hedge h=1, p=6%, phi=22.857143% and H=0.05. Funded reference excess is
6% + 0.05 times 22.857143% = 7.142857%. Using H=0 would lose the cash-aware term.
These deliberately large one-period inputs illustrate the identity, not annual
forecast calibration.

The executable example fits four synthetic stocks from arithmetic basket returns
and reconciles the full asset covariance against independently calculated terminal
asset, forward and reference-loan cash flows. No market-data download is required.

## Implementation in factormodels

Use `FactorCovarEstimator(asset_return_type="simple_payoff", ...)` with one common
factor/response/residual cadence and DIFFERENCE factor increments. Existing
estimator defaults remain LOG. The completed fit records its response coordinate;
`assemble_equity_payoff_risk` rejects a LOG-labelled fit or LOG basket factors.

The API consists of `EquityPayoffRiskSpec`, `EquityPayoffRiskSnapshot`,
`build_funded_currency_exposure`, `compute_payoff_factor_returns` and
`assemble_equity_payoff_risk`. The output owns its matrices, sample dates,
provenance and diagnostics. `to_risk_model()` exports authoritative forecast
covariance and B/H loadings to QIS without inventing a diagonal residual split.

Run the [complete offline example](https://github.com/ArturSepp/FactorModels/blob/main/examples/equity_payoff_risk.py)
with the selected external research environment:

```console
python -m examples.equity_payoff_risk
```

The separate `equity_fx` API and its [original example](https://github.com/ArturSepp/FactorModels/blob/main/examples/equity_fx_risk.py)
remain available as an explicitly first-order local-LOG approximation. They use
`EquityFxRiskSpec`, `EquityFxRiskSnapshot`, `build_currency_exposure` (1-h) and
`augment_equity_fx_model`; they are not aliases for the exact payoff API.
The approved consumer runtime is not upgraded by research validation.

## Interpretation and limitations

Use QIS LOCAL_EXCESS_TRANSLATED / SIMPLE responses, FX_FORWARD / SIMPLE factors
and direct REFERENCE_EXCESS / SIMPLE comparators. Local LOG excess may remain the
signal coordinate; it is not the response of this risk model. Arithmetic factors
need no artificial NAV or log transformation. Optional wealth normalization and
self-financed cash/settlement ledgers are separate contracts.

Implementation acceptance is based on payoff/cash-flow identities, full historical
and fixed-exposure covariance reconciliation, PSD within roundoff tolerance,
input integrity and point-in-time invariance. These correctness tests remain
mandatory. Forecast accuracy is a separate research question.

Consumer holdouts are supplementary and cannot fail implementation acceptance.
Freeze each fit at its cutoff and report the forecast horizon, estimator, sample
and any descriptive reference bands explicitly. Short-sample risk estimates are
noisy; crossing an uncalibrated band is not an implementation defect. Retaining
all joint blocks with a common kernel reproduces direct covariance and does not
create a forecasting improvement by itself. Independent predictive qualification
is separate from this research implementation. Synthetic stocks with observed FX
exercise identities and currency paths without establishing observed-stock
residual dependence or production suitability. A covariance-only QIS risk view
supports volatility, beta and tracking error without discarding correlated risk.

Traded forwards with basis need the additional funding term c = G_local F_market/S
minus G_reference. This API assumes matched synthetic CIP; market-basis inputs,
transaction costs, collateral, funding spreads and execution are not modelled.

## See also

- [Completed factor models](completed_model.md)
- [Canonical API](api.md)
- [Consumer risk handoff](integration.md)

## References

- [FactorModels software citation](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).
- [QIS analytics and software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso estimation and software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).

The payoff assembly and research validation policies describe this implementation;
they are not attributed to a separate published global-equity model.
