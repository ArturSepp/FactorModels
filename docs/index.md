---
myst:
  html_meta:
    description: >-
      factormodels: multi-asset factor models in Python — factor construction, point-in-time loadings and covariance, GLS premia and CMA diagnostics; documentation index.
---

# factormodels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Factor construction, GLS premia, capital market assumptions and factor covariance estimation.

FactorModels implements multi-asset tradable factor models (MATF): construct
factor net asset value histories, estimate asset loadings and covariance, and
analyse forecasts with generalized least squares (GLS) and capital market
assumption (CMA) diagnostics. See the
[package README](https://github.com/ArturSepp/FactorModels) and
[software citation](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Start with the [synthetic reference workflow](reference_workflows.md), then
inspect the [completed model](completed_model.md). The
[GLS/CMA guide](gls_factor_premia.md) connects the fitted model to annual arithmetic
forecasts using an explicit mean bridge and log-risk approximation. Read the Docs hosts
the [online documentation](https://factormodels.readthedocs.io/en/latest/). Committed-source
links identify their stated revision; the local guides describe the checked-out source.

## Installation and documentation versions

The completed-result interfaces in this handbook require **FactorModels 0.3.0**
or later and **Python 3.11** or later:

```console
python -m pip install "factormodels>=0.3.0"
```

The [stable handbook](https://factormodels.readthedocs.io/en/stable/) follows the
released documentation; [latest](https://factormodels.readthedocs.io/en/latest/)
follows the development branch and can include changes after a release. Check
your installed version with `python -m pip show factormodels` and consult the
[changelog](https://github.com/ArturSepp/FactorModels/blob/main/CHANGELOG.md).
The 0.3.0 release removes `.data` and retired deep imports; use the
[migration table](migration.md) when upgrading from 0.2.0.

Library APIs are installed by pip. Scripts under `examples/` belong to the
[source checkout](https://github.com/ArturSepp/FactorModels), so run their commands
from that checkout. The [README quickstart](https://github.com/ArturSepp/FactorModels#five-minute-quickstart)
also supplies its inputs and works with the installed package alone. See
[CONTRIBUTING.md](https://github.com/ArturSepp/FactorModels/blob/main/CONTRIBUTING.md)
for an external build environment; keep environments and generated output outside
OneDrive on Windows. The optional [OP handoff](integration.md) has its own
explicit dependency requirement.

## Handbook

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
gls_contracts
cma_diagnostics
migration
licensing
```

Start with the completed model: definitions and histories, fitted loadings, selected
covariance, diagnostics and provenance. Loading supplied inputs, fitting them,
exporting tables and consuming risk in another package are explicit separate steps.
FactorLasso supplies statistical estimation, OptimalPortfolios consumes alphas
and covariance for optimisation, and QIS supplies portfolio risk and stress
reports. The [integration guide](integration.md) describes these handoffs.
