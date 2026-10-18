# Three reference workflows

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [TradableFactors](https://github.com/ArturSepp/tradablefactors).
Software citation: [CITATION.cff](https://github.com/ArturSepp/tradablefactors/blob/main/CITATION.cff).

The [canonical runnable source](https://github.com/ArturSepp/tradablefactors/blob/main/examples/reference_workflows.py)
offers three input paths. All default execution is offline and load-only. Use
`--fit` to request a completed model and `--output` to persist new outputs.

| Path | Data and completed output | Limitations |
|---|---|---|
| Synthetic | `synthetic_bundle()` supplies 121 monthly USD NAV observations; `run_local_test(1)` returns the fitted model and a separate CMA audit | Teaching data and forecasts, no empirical claims |
| Caller-supplied | `--bundle /path/to/inputs` validates six files; `--fit` explicitly estimates from them | Caller supplies data rights, availability, recipe and conventions |
| Public proxy | `public_proxy_bundle(prices)` constructs Equity=SPY and Rates=TLT long-only factors; `run_local_test(2)` uses synthetic prices under those ticker labels | A two-ETF illustration, not the production MATF recipe; target asset histories are synthetic even when real proxy prices are supplied |

## Reproduce offline

Install the package, keep these example scripts outside site-packages, and run:

```console
python examples/reference_workflows.py
python examples/reference_workflows.py --fit --output /path/to/new_synthetic_run
python examples/reference_workflows.py --bundle /path/to/new_synthetic_run/inputs --fit
```

The first command only constructs/validates synthetic inputs and prints their
count. The second returns an annual fitted covariance and writes input CSVs,
loadings, factor/residual/asset covariance, all CMA table fields (`assets.csv`,
`validation.csv`, etc.), numerical/provenance metadata and a two-page report to a
new directory. The third is the caller-supplied path and writes nothing.

The example independently checks covariance assembly, labels, date bounds and
positive semidefiniteness. The tests block network access, test default load-only
behavior, and inspect real report bytes and tables. Completion does not require
rendering. See [fields and units](completed_model.md).

## Optional public acquisition

Provide a dated CSV with USD adjusted-close columns `SPY` and `TLT`:

```console
python examples/reference_workflows.py --proxy-prices /path/to/proxy_prices.csv --fit
```

Alternatively install `yfinance` separately and explicitly request
`--acquire-proxy`. Acquisition is example-only, never a TF library operation or
an offline-test prerequisite. The example requests adjusted closes for
2014-01-01 through 2024-02-01 (end exclusive), samples month ends, and uses the
overlap with the synthetic target panel. Provider revisions can change those
inputs; retain your acquired CSVs and provenance for reproducibility.

These definitions omit MATF carry, short legs, currency-specific design and
production calibration. The acquired prices are public-provider observations;
the four synthetic target assets and illustrative forecasts remain teaching
inputs. No provider download is necessary to test construction or fitting.

## Inspect or consume

`run_local_test` returns a dictionary containing `bundle`, `snapshot`, `audit` and
`report_status`. The illustrative percentage/bp CMA report requires LOG factors
and positive orthogonal selected residual variances. For native-point factors,
correlated residual policies or zero residual variance, fitting still completes;
`audit` is `None` and `report_status` explains why that report is unavailable.
Tables remain available in memory. `persist_example` is explicit, refuses an
existing output directory, and exports tables rather than a promised checkpoint.
The [separate handoff](integration.md) supplies the completed model to OP and QIS.
