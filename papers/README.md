# Research paper workspaces

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [FactorModels](https://github.com/ArturSepp/FactorModels).
Citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

The maintainer's local workspace holds two companion studies:

| Local directory | Subject | Public status |
| --- | --- | --- |
| `matf_cma_jpm_2026/` | Tradable factors, capital market assumptions and strategic allocation | Entirely ignored; no redistribution approved |
| `achievable_sharpe_faj_2026/` | Achievable Sharpe, factor access and universe selection | Entirely ignored; no redistribution approved |

These directories are absent from public checkouts and package distributions. Their
replication calls FactorModels for factor analytics and OptimalPortfolios for allocation;
provider-specific workflows may also use ROSAA. No consumer dependency is added to FM.

Shared `cma_data` resides privately in this directory. Local settings.yaml files
select it through CMA_DATA_PATH. Immutable snapshots retain their provenance;
MATF-CMA and FAJ pin their numerical vintages independently. This data layer, its
licensed histories and its local production exporter are ignored and excluded
from package distributions. Historical inputs and outputs remain preserved.
See [AGENTS.md](AGENTS.md) for preservation and execution rules.
