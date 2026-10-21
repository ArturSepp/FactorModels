# Three reference workflows

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

The [canonical runnable source](https://github.com/ArturSepp/FactorModels/blob/main/examples/reference_workflows.py)
offers three input paths. All default execution is offline and load-only. Use
`--fit` to request a completed model and `--output` to persist new outputs.

| Path | Data and completed output | Limitations |
|---|---|---|
| Synthetic | Installed `synthetic_bundle()` supplies 121 monthly USD NAV observations; `run_local_test(1)` returns the fitted log-return model, GLS inference and CMA audit | Synthetic arithmetic forecasts use an explicit second-order mean bridge and the unchanged fitted log covariance as a risk approximation |
| Caller-supplied | `--bundle /path/to/inputs` validates six files; `--fit` explicitly estimates from them | Caller supplies data rights, availability, recipe and conventions |
| Public proxy | `public_proxy_bundle(prices)` constructs Equity=SPY and Rates=TLT long-only factors; `run_local_test(2)` uses synthetic prices under those ticker labels | A two-ETF illustration, not the production MATF recipe; target asset histories are synthetic even when real proxy prices are supplied |

## Reproduce offline

The installed `factormodels.synthetic_bundle()` helper supplies fresh inputs
without fitting, acquisition or persistence. The command-line examples live in
the checkout; after installing the package, run from that checkout:

```console
python examples/reference_workflows.py
python examples/reference_workflows.py --fit --output /path/to/new_synthetic_run
python examples/reference_workflows.py --bundle /path/to/new_synthetic_run/inputs --fit
```

The first command only constructs/validates synthetic inputs and prints their
count. The second returns an annual fitted covariance and, for the eligible
synthetic model, GLS inference and a CMA audit. It saves input CSVs, fitted model
tables, forecast and diagnostic tables, provenance and a CMA PDF to a new
directory. The third is the caller-supplied path and writes nothing.

The example independently checks covariance assembly, labels, date bounds and
positive semidefiniteness. The tests block network access, test default load-only
behavior, and check persisted covariance and its units. Completion does not require
rendering. See [fields and units](completed_model.md) and the
[GLS/CMA methodology](gls_factor_premia.md) for the forecast construction and
distinction between calibrated and implied premia. QIS may print a timing line
during currency-adjusted return preparation; it is not a numerical result.

## Optional public acquisition

Provide a dated CSV with USD adjusted-close columns `SPY` and `TLT`:

```console
python examples/reference_workflows.py --proxy-prices /path/to/proxy_prices.csv --fit
```

Alternatively install `yfinance` separately and explicitly request
`--acquire-proxy`. Acquisition is example-only, never a FM library operation or
an offline-test prerequisite. The example requests adjusted closes for
2014-01-01 through 2024-02-01 (end exclusive), samples month ends, and uses the
overlap with the synthetic target panel. Provider revisions can change those
inputs; retain your acquired CSVs and provenance for reproducibility.

These definitions omit MATF carry, short legs, currency-specific design and
production calibration. The acquired prices are public-provider observations;
the four synthetic target assets remain teaching
inputs. No provider download is necessary to test construction or fitting.

## Inspect or consume

`run_local_test` returns a dictionary containing `bundle`, `snapshot`, `audit` and
`report_status`. The canonical synthetic path keeps the fitted loadings and annual
factor, residual and asset covariance unchanged. It starts with synthetic annual
arithmetic factor assumptions of 4% each, applies the explicit mean bridge
described in the [GLS guide](gls_factor_premia.md), and adds asset forecast
perturbations from −0.25% to +0.25%. The full-alpha GLS view credits the constructed
arithmetic alpha. The CMA audit uses the full forecast and records admitted alpha
and the GLS residual as construction channels. Both use zero cash and the fitted
annual log covariance as an explicitly declared risk approximation.

This report adapter supports LOG factors, zero domestic rates and strictly
positive diagonal selected residual covariance. Native-point coordinates,
correlated residuals, zero residual variance and nonzero cash require additional
convention or metric choices. For those cases `audit` is `None` and
`report_status` explains why; the fitted model and its values remain usable.
The full selected covariance is never diagonalized to obtain a report.

`persist_example` is explicit, refuses an existing output directory, and exports
tables rather than a promised checkpoint. Unsupported report cases still export
their complete model tables and status. The [separate handoff](integration.md)
supplies the fitted model to OptimalPortfolios and QIS. The standalone
[arithmetic reference](../examples/gls_factor_premia.py) remains an independent
check of GLS using simple-return inputs.
