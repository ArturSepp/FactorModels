# Contributing

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Use [AGENTS.md](AGENTS.md) for repository conventions and [CITATION.cff](CITATION.cff) for software attribution.

The Windows interpreter is `C:\Python\TradableFactors312\Scripts\python.exe`. Environments, caches, builds and generated reports stay outside OneDrive. Select the repository with the shared `Enter-AgentRepo.ps1` helper before manual checks, or use the registered `Invoke-Repo.ps1 -Repo TradableFactors -Task verify` task.

Keep tests deterministic and offline. Every numerical change needs an independent reference calculation, and a new regression check must detect its controlled counterexample. Preserve input units, dates, labels and selected residual policy. Compare changed APIs through their numerical outputs; do not restore removed OP interfaces.

Commit source, tests and documentation. Do not commit local `agents/` records, production bundles, market-data snapshots, generated reports or manuscript artifacts. Public distribution waits for the documented rights and MIT dependency gates.
