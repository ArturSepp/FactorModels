# Supplied six-file bundles

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

`load_bundle(directory)` is the installed load-only API. The
[canonical reference script](https://github.com/ArturSepp/FactorModels/blob/main/examples/reference_workflows.py)
accepts `--bundle` and fits only when `--fit` is supplied. No market-data provider
or OP installation is needed.

## One explicit schema

| File | Contents |
|---|---|
| `futures_risk_factors.csv` | Dated factor NAVs or native observed levels, in configured factor order |
| `fx_hedging_data_fx_spots.csv` | Dated USD per currency-unit spots, including USD=1 anchor |
| `fx_hedging_data_domestic_rates.csv` | Dated annual decimal domestic rates for all required currencies |
| `asset_prices.csv` | Dated positive asset prices in each asset's local currency |
| `asset_metadata.csv` | Asset-indexed `currency`, `hedge_ratio` in [0,1], and `return_frequency` |
| `risk_model_settings.csv` | One `value` column indexed by settings name, including `schema_version=1` |

Create settings with `BundleSettings` and persist with `save_bundle`; its
`to_frame()` emits every required field. `from_frame()` rejects missing or unknown
settings and unsupported versions. JSON-valued cells preserve named factor maps
and cadence parameters. Unversioned historical bundles are not accepted by this
API. Convert active inputs in a separate working copy and retain the original.

## Units, configuration and dates

The settings state reference currency, ordered factors, recipe identity,
LOG/DIFFERENCE factor coordinates and native units, total/excess basis, FX quote
and rate units, covariance units, factor-return cadence/span, rolling cadence,
FL model/penalty/loss normalization, beta span/warmup, demeaning, cash-rate lag,
zero-return handling, optional cadence
overrides and the selected residual policy. Changing a default constructor value
does not reinterpret an already serialized schema-v1 settings file.

`estimation_end` is the information cutoff. `estimation_start` selects the first
rolling output date; earlier observations remain warmup history. Current fitting
uses history through the cutoff. Before FX conversion and fitting, all histories
are clipped to that cutoff. Input timestamps and declared units are caller
assertions, not independent provider evidence.

The [QIS reference-currency conversion and excess-returns guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/reference_currency_excess_returns.md)
owns the conversion and reference-cash calculation contract. Its
[USD/CHF worked example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/reference_currency_excess_returns.md#the-same-usd-asset-in-usd-and-chf-excess-returns-and-betas)
compares hedge ratios 0, 0.5 and 1 for the same asset against a fixed USD factor.
See also [why fully hedged payoffs can differ](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/reference_currency_excess_returns.md#why-full-hedging-does-not-reproduce-the-usd-payoff-exactly)
and [why a mean excess-return difference alone does not change beta](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/reference_currency_excess_returns.md#why-a-different-mean-excess-return-alone-does-not-change-beta).
The USD 100-to-110 illustration shows that an opening-principal hedge leaves FX
exposure on the investment gain. The beta identity uses equal-weight OLS with an
intercept on a common sample; it is a teaching example, not a claim of invariance
for every FactorModels estimator. Preserve the chosen factor basis, observation
sample, mean adjustment and regularisation when interpreting a fitted model.

## Load, fit, persist

`load_bundle` returns owned input tables and source file SHA-256 hashes. It does
not estimate a model or modify files. `fit_bundle(bundle)` returns one
`FactorModelSnapshot`; `fit_bundle(bundle, rolling=True)` returns dated snapshots
in a `RollingFactorModel`. The complete estimator is the same `FactorCovarEstimator` pipeline:
QIS currency/return preparation and factor covariance, followed by FL estimation
and the selected residual assembly.

`save_bundle(bundle, new_directory)` writes the six inputs, never replacing an
existing directory. It does not save a fitted checkpoint or arbitrary in-memory
`FactorBundle.provenance` entries; keep those separately when needed. Reloading
records the six source files' hashes. The reference script
separately exports inspectable result tables and a report only on request.

No runtime historical-format adapter, provider fetching, hidden refit or
overwrite mode is supplied. See [completed fields](completed_model.md) and
[downstream usage](integration.md).
