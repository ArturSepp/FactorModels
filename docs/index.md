# TradableFactors documentation

*Author: [Artur Sepp](https://github.com/ArturSepp)*

TradableFactors provides completed factor-model and GLS/CMA diagnostic results. See the [package README](https://github.com/ArturSepp/tradablefactors) and [software citation](../CITATION.cff).

```{toctree}
:maxdepth: 2

api
completed_model
reference_workflows
rolling_factor_covar_from_csv
integration
construction
factor_covariance_hcgl
gls_factor_premia
cma_diagnostics
migration
```

Start with the completed model: definitions and histories, fitted loadings, selected
covariance, diagnostics and provenance. Loading supplied inputs, fitting them,
exporting tables and consuming risk in another package are explicit separate steps.
The installed bundle workflows are available; the active paper/Rosaa cutover remains
a later migration stage.
