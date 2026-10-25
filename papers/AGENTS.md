# Private paper workspaces

The repository instructions control package ownership and external environments.
MATF-CMA (`matf_cma_jpm_2026`) and FAJ (`achievable_sharpe_faj_2026`) are entirely
local, ignored research workspaces. Neither manuscript, replication code, licensed
input, presentation nor private correspondence is approved for redistribution.
Never force-add them. Both workspaces are excluded from wheels and source archives.

Preserve the existing paper, drafts, presentations, private, replication and agents
sections, including historical layouts and immutable evidence. Existing paper-specific
agent records remain private; repository-wide migration records belong in root agents/.
Do not rewrite frozen archives, numerical outputs, seeds or historical provenance.

Shared `cma_data` now resides under FactorModels/papers/cma_data, following the
25 October 2026 authorised private relocation and model-CMA refresh. Both papers
use their ignored settings.yaml CMA_DATA_PATH. Keep immutable historical cuts
unchanged; new calibrations receive new tags. Do not duplicate the shared source
inside either paper. MATF-CMA uses the latest Q3 model cut without House View;
FAJ retains its existing numerical cut until its separately requested refresh.
Optimisation calls remain in OptimalPortfolios. Paper workflows may consume OP and
ROSAA; FactorModels library code must not import those consumers or paper modules.

Run the two replication suites in separate processes from C-local source exports.
The approved composed interpreter for this relocation is
C:/Python/FactorModelsPapers312/Scripts/python.exe; use it read-only and preserve all
historical environments. Dot-source Enter-AgentRepo.ps1 for FactorModels first, route
cache and temporary output to AGENT_LOCAL_ROOT, and never rebuild exhibits in OneDrive.
Record baseline failures separately from relocation regressions and compare immutable
file hashes before and after. A folder move does not authorise a numerical refresh.
