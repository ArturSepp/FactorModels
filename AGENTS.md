# FactorModels contributor instructions

## Scope and package ownership

FactorModels (FM) owns factor construction, complete fitted factor models,
generalized least squares (GLS), capital market assumption (CMA) analytics and
method-specific reports. It uses QuantInvestStrats (`qis`) and FactorLasso for
shared analytics and statistical estimation. OptimalPortfolios consumes supplied
alphas and covariance; application-specific recipes, providers and scheduling
belong to downstream consumers. The library must not import those consumers.

Follow the current task's explicit scope. Historical migration decisions are not
blanket permission for new public-interface or numerical changes. Keep the
canonical `factormodels` interface without retired-package forwarding modules or
an OptimalPortfolios compatibility extra. Numerical changes require explicit
authorisation and independent reference evidence.

## Environment and outputs

Use the portable source, lint, test, build and documentation commands in
[CONTRIBUTING.md](CONTRIBUTING.md). Work from an external checkout for builds and
keep environments, caches, reports and temporary output outside synchronised
folders. Select the declared dependency groups and use the reviewed lockfile.

On the registered maintainer Windows/OneDrive workspace, use
`C:\Python\FactorModels312\Scripts\python.exe`. Never create or use a virtual
environment, cache, build or temporary output tree inside OneDrive. Before manual
Python or check commands, dot-source the workspace
`ArturSepp/scripts/repo_governance/Enter-AgentRepo.ps1` for this repository. The
registered launcher provides `check`, `test`, `verify` and `docs` tasks. Generated
state belongs under the external `AGENT_LOCAL_ROOT`; project `uv` operations must
set `UV_PROJECT_ENVIRONMENT` to the external environment explicitly. Preserve any
frozen historical environment and use a separate environment for candidate work.

## Public API and tests

Use Google-style docstrings, explicit module exports and root `__all__`. The root
public export list is the canonical package application programming interface
(API). Test canonical objects and their contracts. Keep deterministic synthetic
fixtures free of private data and provider dependencies. Source-adjacent tests
are shipped for installed-wheel checks; top-level tests hold package acceptance
contracts. Examples use an explicit `run_local_test(case)` entry point and run
offline unless they explicitly describe an optional acquisition action.

Every fitted result states cutoff, ordering, return transformations, annualisation,
currency when supplied, selected residual policy and relevant provenance. The
`qis.RiskModel` is an interchange view, not the complete fitted result. Preserve
full selected covariance; label diagonal-reference decompositions explicitly.

Read the [shared documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md)
before substantive reader-facing documentation edits. Maintain `LICENSE`,
supplemental notices, `CITATION.cff`, the README citation and `CHANGELOG.md`.
Version metadata must agree. Keep generated analyses and private inputs out of
source distributions and the public repository.

## Generated shared instructions

The block below is generated from the maintainer's shared instruction source.
Amend that source and run its scoped producer; do not hand-edit the block or
change unrelated repository emissions. The current task's authorisation defines
which files and public-interface changes are in scope.

<!-- ===== SHARED AGENT CORE (consumer variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-10-20, agent core v1.6; scoped FactorModels amendment -->

## Domain invariants

Not inferable from any single file, and the source of numerically wrong code that runs clean:

- **No look-ahead, anywhere in a backtest path.** A weight decided at *t* is applied over
  *[t, t+1]*. Estimation is point-in-time: `MeanAdjType.INSAMPLE` subtracts a full-sample mean
  and is therefore forward-looking — correct for a descriptive exhibit, wrong inside a backtest.
- **Return convention is stated, never implied** — `qis.to_returns(..., is_log_returns=...)`.
  Annualisation follows from the frequency; never silently switch convention, frequency, or
  annualisation factor.
- **Sharpe has three explicitly labelled conventions** in `qis`; excess variants need
  `PerfParams.rates_data`. State which one a number uses.
- **`qis.BootstrapType.STATIONARY` wraps circularly from qis 5.1.0.** Any result resampled under
  an earlier version does not reproduce.
- One convention per concept across the stack. If two packages disagree, that is a bug to
  report, not a difference to accommodate.

## Use the stack before you write it

This package consumes `qis` (analytics, backtesting, risk, stress testing and reporting) and
`factorlasso` (statistical estimation and fitted-data containers). FactorModels owns the
complete factor construction and estimation pipeline that connects them. Its finite outputs
are factor definitions, net asset value (NAV) histories and fitted loadings/covariance for an asset universe; document
which construction stages are available and each object's downstream usage. OptimalPortfolios consumes
explicit alphas and covariance for optimisation. Reimplementing an upstream capability is
a defect, not a convenience.
Triggers — stop and check the export list before writing: backtest, rebalance, turnover,
drawdown, Sharpe, volatility target, bootstrap, resample, unsmooth, covariance, correlation,
regime, hedge ratio, factsheet, tracking error, realised tracking error, information ratio,
benchmark beta, marginal contribution, risk contribution.

- **The hard stop:** a `for` loop over dates accumulating a position, a weight or a P&L is
  `qis.backtest_model_portfolio`. The hand-rolled version gets drift adjustment wrong — `qis`
  holds *units* between rebalancings, not weights.
- **Never hand-roll `d' Σ d`, a beta ratio, or a tracking-error (TE) decomposition.** Ex-ante tracking error,
  factor exposures, benchmark beta and marginal TE come from `qis.RiskModel`, built from
  fitted FactorModels output via `factormodels.build_risk_model`; supplied covariance uses
  `qis.RiskModel(covar={date: covariance})`. Realised ex-post TE is
  `qis.compute_ewma_realised_tracking_error` (EWMA series), and whole-sample TE/IR scalars are
  `qis.compute_te_ir_errors`.
- **Never invent a symbol.** If a function, class, or keyword argument is not in the export
  list, it does not exist. Check in one line —
  `python -c "import qis; print([n for n in dir(qis) if 'unsmooth' in n.lower()])"`;
  `qis.api.CORE_API` is the documented core and `help(qis.<symbol>)` gives the arguments. Say a
  symbol is missing rather than producing code that calls it.
- **If you genuinely must reimplement**, name the rejected stack symbol and why, in a comment on
  the line above the definition — that turns a silent divergence into a reviewable decision.
- Never introduce `quantstats`, `pyfolio`, `empyrical`, `ffn`, `bt`, or an ad-hoc statistics
  layer.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: estimation windows, weight normalisation, annualisation, constraint construction,
  anything resampled. Verify against a reference computed a different way, and say which.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Agent-generated artifacts

All agent-generated roadmaps, execution plans, audits, reports, handoffs, and other working
outputs live under the repository-root `agents/` directory, which is local and ignored by Git.
Never create `ROADMAP_*.md`, `Claude outputs/`, `Codex outputs/`, or similar agent-output
artifacts at the repository root. Name feature roadmaps `agents/ROADMAP_<feature>.md`. An
execution request names the file and stage. A stage is complete when its stated verification
command passes; its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->
