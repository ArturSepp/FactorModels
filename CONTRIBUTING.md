# Contributing

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Contribute to [FactorModels](https://github.com/ArturSepp/FactorModels) using a
reviewed source checkout and the dependency versions in `uv.lock`. See
[AGENTS.md](AGENTS.md) for numerical and package boundaries and
[CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff) for software attribution.

## Prepare an external checkout

Use Python 3.12 and `uv` for the reference contributor commands below. The package
supports Python 3.11 and later; continuous integration (CI) checks its declared
platform/interpreter matrix separately. Obtain an authorised checkout of the
revision you intend to test, and run these commands from its root. A source
archive also works for the direct checks below; Git-based shared checkers require
a checkout.

Keep this working checkout outside OneDrive or other synchronised folders.
Builders can write metadata beside source files, so redirecting only the final
artifact directory is insufficient. Choose one setup block for your shell; it
places environments, caches and generated results outside the checkout.

On Linux or macOS:

```sh
export FM_SOURCE="$PWD"
export FM_WORK="$(mktemp -d)"
export UV_PROJECT_ENVIRONMENT="$FM_WORK/venv"
export UV_CACHE_DIR="$FM_WORK/cache/uv"
export PYTHONPYCACHEPREFIX="$FM_WORK/cache/python"
export NUMBA_CACHE_DIR="$FM_WORK/cache/numba"
export MPLCONFIGDIR="$FM_WORK/cache/matplotlib"
export MPLBACKEND=Agg
export PYTEST_ADDOPTS="-o cache_dir='$FM_WORK/cache/pytest' --basetemp='$FM_WORK/tests'"
```

On Windows, use PowerShell and an external environment under `C:\Python`:

```powershell
$env:FM_SOURCE = (Get-Location).Path
$env:FM_WORK = Join-Path $env:LOCALAPPDATA ('FactorModels\contributor-' + [guid]::NewGuid().ToString('N'))
$env:UV_PROJECT_ENVIRONMENT = 'C:\Python\FactorModelsContributor312'
$env:UV_CACHE_DIR = Join-Path $env:FM_WORK 'cache\uv'
$env:PYTHONPYCACHEPREFIX = Join-Path $env:FM_WORK 'cache\python'
$env:NUMBA_CACHE_DIR = Join-Path $env:FM_WORK 'cache\numba'
$env:MPLCONFIGDIR = Join-Path $env:FM_WORK 'cache\matplotlib'
$env:MPLBACKEND = 'Agg'
$env:PYTEST_ADDOPTS = "-o cache_dir='$($env:FM_WORK)/cache/pytest' --basetemp='$($env:FM_WORK)/tests'"
```

Use a different `C:\Python` environment name if that contributor environment is
already in use. Do not sync an unrelated or frozen historical environment.

## Source checks and documentation

After the setup block, these commands work in either shell:

```console
uv sync --locked --python 3.12 --group test --group lint --extra docs
uv run --no-sync ruff check src tests tools
uv run --no-sync interrogate
uv run --no-sync deptry src
uv run --no-sync pytest
```

The `test` and `lint` dependency groups supply the reviewed tools; the `docs`
extra supplies Sphinx and its extensions. `--locked` rejects a stale lockfile
instead of changing the dependency cohort. Keep intentional dependency updates
separate from unrelated source changes.

The source test suite runs `tools/check_architecture.py` through
`tests/test_architecture.py`. It checks absolute, relative, nested and typing-only
imports against the ownership matrix, resolves package re-exports and rejects
implementation imports through the FM root facade. Standard-library imports are
allowed separately from numerical dependencies. Module-initialization cycles fail;
the model/risk deferred cycle is listed explicitly.

The final boundary is enforced without transition permissions: GLS/CMA and
opportunity/uncertainty live in `gls_diagnostic`. Its only FactorLasso imports are
the four named calibration functions in the three named modules. Estimation owns
FL fitting and materialization. The local-method cycle between completed models
and risk export remains permitted because neither side needs the other during
module initialization. Shipped tests live beside their component and use
`tests/test_*.py`; package-wide acceptance checks can remain in the shared tests.

Production signatures select Ruff `ANN001`, `ANN002`, `ANN003`, `ANN201`, `ANN202`
and `ANN204`. Every production signature is annotated; there are no temporary
production exceptions. `tools/architecture_debt.json` retains empty annotation
and header inventories, and the tests require them to remain empty.
Tests (including their initializers), examples and tooling are exempt from these
six annotation rules only. Correctness and import rules continue to apply.

Numerical module headers must contain substantive `Purpose:`, `Inputs:`, `Outputs:`,
`Units:`, `Conditioning:` and `Ownership:` fields. These fields describe the actual
mathematical contract; automated checks cannot establish semantic correctness.
Export-only initializers and tests are exempt; the path resolver and version lookup
have named infrastructure exemptions. Every numerical implementation module complies;
new modules must comply immediately. Keep controlled-defect tests for the guards.

Estimation orchestration lives in `_pipeline`, with cadence fitting in `_frequency`,
cluster schedules in `_clustering` and residual preparation/reuse in `_residuals`.
These private stages preserve the estimator's operation order and FL/QIS ownership.
Both CMA renderers use `reports._pdf` for atomic export, figure closure and failure
cleanup; `utils.local_path` remains responsible only for settings and path resolution.

Build the source distribution and then its wheel with the default `build`
sequence, and build HTML with warnings treated as errors. On Linux or macOS:

```sh
uv run --no-sync python -m build --outdir "$FM_WORK/dist"
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs "$FM_WORK/html"
```

The equivalent PowerShell commands are:

```powershell
uv run --no-sync python -m build --outdir "$env:FM_WORK\dist"
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs "$env:FM_WORK\html"
```

## Installed-artifact checks

An editable source install does not verify wheel contents. Install the newly
built wheel into a separate environment and run from an empty directory, using
its absolute Python path. For a reproducible acceptance run, export the locked
runtime and test requirements with `uv export --locked --group test --no-dev
--no-emit-project`, install that export, then install the wheel with `--no-deps`.
Do not use `uv run` from the source project for the installed check: it selects
the project's editable environment.

Run `python -m pytest --pyargs factormodels` with that installed interpreter.
Also pass the absolute source paths for `tests/test_examples.py`, `tests/test_readme.py`, `tests/test_documented_contracts.py` and
`tests/test_dependency_floor.py`. Those top-level acceptance contracts complement
the shipped source-adjacent tests; their collection counts need not match.
Confirm `factormodels.__file__` resolves inside the installed environment.
The wheel job in [.github/workflows/ci.yml](.github/workflows/ci.yml) records the
installed execution contract.

## Numerical and contribution rules

Keep tests deterministic and offline. Every numerical change needs an independent
reference calculation, and a new regression check must detect its controlled
counterexample. Preserve input units, cutoffs, dates, labels, seeds, return
transformations, annualisation and the selected residual policy. Compare changed
interfaces through their numerical outputs. The library must not import its
optimisation or application consumers.

Offline examples require neither acquisition services nor an optimiser package.
Public data acquisition and the OptimalPortfolios composed example are explicit,
separate actions. Use the runnable sources linked from the
[reference workflows](docs/reference_workflows.md).

Contribute source, tests and documentation. Keep local `agents/` records,
production inputs, market-data snapshots, generated reports and manuscripts out
of Git and distributions. Preserve the source licence and supplemental notices;
see the [licensing guide](docs/licensing.md).

## Registered maintainer workspace

On the maintainer's Windows/OneDrive checkout, use
`C:\Python\FactorModels312\Scripts\python.exe` and the registered
`ArturSepp/scripts/repo_governance/Enter-AgentRepo.ps1` helper before manual Python
or check commands. The helper sets the external `AGENT_LOCAL_ROOT` and routes
caches and temporary state to local storage. Never create or use a virtual
environment inside OneDrive, and never run a project `uv` command there without
an explicit external `UV_PROJECT_ENVIRONMENT`.

Use the registered `Invoke-Repo.ps1 -Repo FactorModels -Task verify` or `-Task docs`
launcher when that maintainer workspace is available. The shared documentation
checker validates the staged selection by default; for unstaged work use
`.github/oss_checks.py docs --working-tree --python <external-python>` with an
external `--output-dir`, after the helper. Builds use the existing external
source-export path. These registered helpers are local conveniences, not
prerequisites for the direct contributor commands above.
