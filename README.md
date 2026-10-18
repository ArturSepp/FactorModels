# TradableFactors

*Author: [Artur Sepp](https://github.com/ArturSepp)*

TradableFactors constructs factor NAVs from supplied prices and definitions, fits factor models, and provides generalized least-squares (GLS), Sharpe-accounting and capital-market-assumption (CMA) analytics. Software citation: [CITATION.cff](CITATION.cff).

## Development status

This is the implementation checkpoint for migration stages 0–6 (`0.1.0.dev3`). It contains factor construction, numerical analytics, the complete in-memory estimation pipeline and a versioned six-file bundle API. Active paper and Rosaa consumer cutover remains Stage 7. This checkpoint is not a PyPI release.

## Package boundary

Construction produces a model specification, an ordered NAV/level panel and optional loading constraints. These are usable inputs to estimation without Rosaa. The fitted product contains loadings, factor and residual covariance, selected asset covariance, diagnostics and dated provenance. FactorLasso supplies statistical estimation. Existing QIS interfaces consume risk views and render portfolio/stress reports. OptimalPortfolios consumes supplied alphas and covariance; TradableFactors never imports the optimiser.

GLS/CMA diagnostics are completed numerical results in their own right. Rendering is optional and stays beside these methods in `tradablefactors.reports`, using QIS plotting primitives. Proprietary recipes, market-data providers and production scheduling remain in Rosaa.

## Local installation

Use a Python environment outside OneDrive:

```console
python -m pip install -e .
python -m pytest
```

The [contributor instructions](CONTRIBUTING.md) describe the registered Windows environment and external output locations. Start with the [completed-output contract](docs/completed_model.md), [three reference paths](docs/reference_workflows.md), and [construction workflow](docs/construction.md). The canonical numerical API is documented in [docs/api.md](docs/api.md); the direct migration contract is in [docs/migration.md](docs/migration.md).

The [canonical example](examples/reference_workflows.py) defaults to loading only:

```console
python examples/reference_workflows.py
python examples/reference_workflows.py --fit
python examples/reference_workflows.py --bundle /path/to/supplied_bundle --fit
```

Add `--output /path/to/new_directory` explicitly to export tables, input CSVs and a
method-specific PDF. No fitting, acquisition or output replacement happens on load.
The completed model can be inspected without an alpha forecast or portfolio solve;
[downstream usage](docs/integration.md) is a separate step.

## Licensing

First-party source is MIT; original notices are retained. Dependencies have their own licences. The current development dependency floor is not a claim of a fully permissive resolved installation: the verified MIT FactorLasso release and dependency-floor check remain public-release gates.

## Citation

```bibtex
@software{sepp_tradablefactors,
  author = {Artur Sepp},
  title = {tradablefactors: Tradable factor models and GLS analytics},
  version = {0.1.0.dev3},
  url = {https://github.com/ArturSepp/tradablefactors},
  year = {2026}
}
```
