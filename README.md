# TradableFactors

*Author: [Artur Sepp](https://github.com/ArturSepp)*

TradableFactors constructs factor NAVs from supplied prices and definitions, fits factor models, and provides generalized least-squares (GLS), Sharpe-accounting and capital-market-assumption (CMA) analytics. Software citation: [CITATION.cff](CITATION.cff).

## Development status

This is the implementation checkpoint for migration stages 0–4 (`0.1.0.dev1`). It contains generic factor construction, numerical analytics and the in-memory factor-estimation pipeline. The installed bundle loader and full downstream migration have separate stage gates. It is not a production cutover or PyPI release.

## Package boundary

Construction produces a model specification, an ordered NAV/level panel and optional loading constraints. These are usable inputs to estimation without Rosaa. The fitted product contains loadings, factor and residual covariance, selected asset covariance, diagnostics and dated provenance. FactorLasso supplies statistical estimation. Existing QIS interfaces consume risk views and render portfolio/stress reports. OptimalPortfolios consumes supplied alphas and covariance; TradableFactors never imports the optimiser.

GLS/CMA diagnostics are completed numerical results in their own right. Rendering is optional and stays beside these methods in `tradablefactors.reports`, using QIS plotting primitives. Proprietary recipes, market-data providers and production scheduling remain in Rosaa.

## Local installation

Use a Python environment outside OneDrive:

```console
python -m pip install -e .
python -m pytest
```

The [contributor instructions](CONTRIBUTING.md) describe the registered Windows environment and external output locations. Start with the [construction workflow](docs/construction.md) and its [offline runnable example](examples/construct_factors.py). The canonical numerical API is documented in [docs/api.md](docs/api.md); the direct migration contract is in [docs/migration.md](docs/migration.md).

## Licensing

First-party source is MIT; original notices are retained. Dependencies have their own licences. The current development dependency floor is not a claim of a fully permissive resolved installation: the verified MIT FactorLasso release and dependency-floor check remain public-release gates.

## Citation

```bibtex
@software{sepp_tradablefactors,
  author = {Artur Sepp},
  title = {tradablefactors: Tradable factor models and GLS analytics},
  version = {0.1.0.dev1},
  url = {https://github.com/ArturSepp/tradablefactors},
  year = {2026}
}
```
