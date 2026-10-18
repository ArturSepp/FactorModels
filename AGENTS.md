# TradableFactors contributor instructions

## Scope and package ownership

TradableFactors owns tradable-factor construction, complete fitted factor models,
GLS/CMA analytics and method-specific reports. It uses QIS and FactorLasso. It must
not import OptimalPortfolios, Rosaa or other consumer packages in its library.
Rosaa retains proprietary recipes, providers and scheduling. OP consumes supplied
alphas/covariances. QIS remains unchanged by this migration.

The active owner-approved roadmap is agents/ROADMAP_TRADABLEFACTORS_OSS_MIGRATION_ASTRA.md.
The clean-migration decision permits removing old APIs and changing signatures;
no legacy shims or OP compatibility extra are required. Stages are development
checkpoints, not public releases. Numerical changes outside explicit cutoff
semantics still require independent reference evidence.

## Environment and outputs

Use C:\Python\TradableFactors312\Scripts\python.exe. Never create an environment,
cache, build, generated report or temporary tree in OneDrive. Before Python/checks,
use the workspace ArturSepp/scripts/repo_governance/Enter-AgentRepo.ps1 with this
repository. The registered launcher runs check, test and verify tasks. Generated
state goes under the external AGENT_LOCAL_ROOT.

## Public API and tests

Use Google-style docstrings, explicit module exports and root __all__. The root
public export list is the canonical package API. Test canonical objects and their
new contracts, not removed OP imports. Keep deterministic synthetic fixtures free
of private data and provider dependencies. Source-adjacent tests are shipped for
installed-wheel checks; top-level tests hold package acceptance contracts.
Examples use an explicit run_local_test(case) entry point and run offline.

Every fitted result states cutoff, ordering, return transformations, annualisation,
currency when supplied, selected residual policy and relevant provenance. QIS
RiskModel is an interchange view, not the complete fitted result. Preserve full
selected covariance; label diagonal-reference decompositions explicitly.

Read the shared documentation standard in ArturSepp/docs/documentation_standard.md
before substantive reader-facing documentation edits. Maintain LICENSE and notices,
CITATION.cff, README citation and CHANGELOG.md. Version metadata must agree.

## Shared-core reconciliation

The consumer block below is generated from the scoped shared-master amendment.
Shared code dispatch and OSS instructions v2.7 now record TF ownership and MIT;
research instructions v2.8 carry the same code core. Only this repository's
consumer block is regenerated; unrelated repository emissions stay at their
recorded hashes. The exact source and target hashes are recorded in stage0.json.
For this authorised migration, the documented stage scope already covers the
multi-file and public-signature work described by the shared escalation rule.

<!-- ===== SHARED AGENT CORE (consumer variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-10-18, agent core v1.6; scoped TradableFactors amendment -->

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
`factorlasso` (statistical estimation and fitted-data containers). TradableFactors owns the
complete factor construction and estimation pipeline that connects them. Its finite outputs
are factor definitions, NAVs and fitted loadings/covariance for an asset universe; document
which construction stages are available and each object's downstream usage. OP consumes
explicit alphas and covariance for optimisation. Reimplementing an upstream capability is
a defect, not a convenience.
Triggers — stop and check the export list before writing: backtest, rebalance, turnover,
drawdown, Sharpe, volatility target, bootstrap, resample, unsmooth, covariance, correlation,
regime, hedge ratio, factsheet, tracking error, realised tracking error, information ratio,
benchmark beta, marginal contribution, risk contribution.

- **The hard stop:** a `for` loop over dates accumulating a position, a weight or a P&L is
  `qis.backtest_model_portfolio`. The hand-rolled version gets drift adjustment wrong — `qis`
  holds *units* between rebalancings, not weights.
- **Never hand-roll `d' Σ d`, a beta ratio, or a TE decomposition.** Ex-ante tracking error,
  factor exposures, benchmark beta and marginal TE come from `qis.RiskModel`, built from
  fitted TF output via `tradablefactors.build_risk_model`; supplied covariance uses
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
