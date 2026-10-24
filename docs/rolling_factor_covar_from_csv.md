<a id="supplied-six-file-bundles"></a>

# Factor estimation from supplied CSVs

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Load ordinary DataFrames with pandas and call `FactorCovarEstimator` directly.
The development version removes the six-file input API; it does not impose a
replacement serialization schema. Historical experiments keep their original
files and recorded package environments. See the [migration guide](migration.md).

<a id="one-explicit-schema"></a>

## Supplied histories

The [reference script](../examples/reference_workflows.py) accepts two CSV paths:

```console
python examples/reference_workflows.py --factor-prices factors.csv --asset-prices assets.csv --fit
```

Both CSVs have a dated index and one numeric column per factor or asset. This
example requires monthly USD total-return NAVs/prices. Factor column order is
retained; asset columns determine the output asset order. It prepares decimal
log asset returns with `qis.to_returns(..., is_log_returns=True)`, preserves the
initial missing return, and treats exact zero returns as missing, then calls
`FactorCovarEstimator.fit_current` with an explicit cutoff. The example's LOG
factor configuration requires positive NAVs. Validate source quality and retain
the recipe, input units and source provenance yourself.

Relative CSV paths use YAML `RESOURCE_PATH`; an explicit `--output` uses
`OUTPUT_PATH`. Absolute paths retain their destination. See
[local path configuration](reference_workflows.md).
Loading alone never fits or writes. `--fit` estimates in memory; `--output`
exports model and eligible report tables to a new directory and refuses overwrite.

<a id="units-configuration-and-dates"></a>

## Configure the estimator explicitly

The [minimal example](../examples/minimal_model.py) demonstrates the complete
definition, construction, rolling estimation and covariance retrieval workflow.
Supply factor NAVs or declared native levels as `risk_factor_prices`, and prepared
decimal log asset returns as `asset_returns_dict`, keyed by cadence. Configure
`LassoModel` and `FactorCovarEstimator` in Python: factor-covariance frequency and
span, regression span and penalty, rolling cadence, coordinate map and selected
residual policy are explicit choices. There is no settings CSV parser.

`fit_current(..., cutoff=...)` returns a `FactorModelSnapshot`;
`fit_rolling(..., time_period=...)` returns a `RollingFactorModel` whose
`covariances` map is keyed by fitted dates. Earlier history remains available for
warmup; each fit truncates observations at its own cutoff. Covariance is annual
in the declared coordinates. Factor and asset total/excess bases must agree.
Input dates do not independently prove provider availability or publication delays.

<a id="load-fit-persist"></a>

## Currency preparation and persistence

For mixed currencies, hedging or excess-return preparation, use
`qis.FxRatesData.compute_fx_adjusted_returns(..., is_log_returns=True)` explicitly
before estimation. Supply the correct FX quote/rate conventions, currency labels,
hedge ratios, cash lag and observation cadences, and clip all preparation inputs
at the information cutoff. The USD price example performs none of these adjustments.
See [QIS](https://github.com/ArturSepp/QuantInvestStrats) and its
[software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

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

Persist input tables with pandas or your application's storage layer. Retain
configuration and provenance separately. Result-table exports are inspectable
tables, not a fitted-object checkpoint format. See
[completed fields](completed_model.md) and [downstream usage](integration.md).
