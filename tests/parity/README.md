# Numerical migration reference

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/tradablefactors/commit/2e23f5a9334fe2c44a5acfa028e0ac3e368f93a0)*

Implemented in [tradablefactors](https://github.com/ArturSepp/tradablefactors).

`manifest.json` describes the fixed reference cohort, tolerances, required
artifacts and paper test inventory. It contains safe identifiers and hashes,
not private numerical data. `tools/replay_reference.py` captures an immutable
reference and compares its labelled numerical content with canonical TF APIs.

The maintainer's local `agents/stage0.json` (or the file named by
`TRADABLEFACTORS_STAGE0`) supplies the private frozen-source/input mapping and
reference interpreter identities. A missing mapping or artifact is an error;
the runner does not fetch data or substitute synthetic data for a paper result.
Public package regression tests use separate deterministic synthetic fixtures.

The CLI accepts `--mode capture|verify`,
`--implementation reference|tradablefactors|consumer`, `--case all|papers|rosaa`,
`--manifest PATH` and `--directory PATH`. Capture is reference-only and refuses
to overwrite its output. Verify does not change the baseline. The `papers` and
`rosaa` checks can select their section of the established `all.json` baseline.
Consumer paper replay additionally requires `--consumer-op-root PATH`, naming
the explicit active research checkout. `--receipt PATH` writes a new external
JSON record of imported consumer sources, their SHA-256 identities and the
full actual report data. An existing receipt is never replaced.

Run historical references in the preserved environment against the frozen old
source. Run TF/consumer checks in an explicit composed environment if checking
OP optimisation and private Rosaa construction as well as TF. These are
integration checks; OP and Rosaa are not TF core dependencies. Keep generated
state, environments and numerical outputs outside the source checkout.

`tradablefactors` mode deliberately holds the unchanged downstream reference
consumers fixed while checking the extracted estimator/analytics. `consumer`
mode executes FAJ's `paper_inputs.load_production_inputs`, MATF-CMA's
`cma_diagnostics_adapter.diagnose_paper_snapshot`, and Rosaa's production
`panel_io.update_factor_panel`. The supplied saved-fit paths are the frozen
cuts recorded in stage 0; no refit is performed. FAJ's prepared numerical inputs
are compared through the unchanged diagnostic projection. MATF-CMA's actual
report is also checked against that projection for the common GLS fields.
Its additional admission, channel and scenario outputs are recorded in the
receipt; the old minimal baseline did not cover those fields.

Rosaa's construction replay injects the original synthetic price fixture at
`load_base_futures_prices` and captures the resulting panel at `save_factor_panel`.
Its production assembly and TF construction execute unchanged between those two
IO seams. This is a consumer construction regression, not a claim that a current
production data refresh or full production shadow run occurred. Consumer mode
rejects an active source location inside the frozen reference tree.

The local frozen `manifest.json` remains byte-identical to the reference.
Git checks out its text with LF line endings; the original local receipt used
CRLF. Operational current/historical paper-test classification belongs in
`consumer_manifest.json`, which retains `reference_manifest_sha256` for the
original bytes and records `reference_manifest_lf_sha256` for the Git text.
Inventory validation binds both digests to the same content in those two exact
line-ending forms. Changed tolerances, labels or JSON formatting fail validation,
as do identity fields that do not match those forms. All other artifact hashes
remain raw byte hashes; the numerical baseline and frozen receipt identities
are unchanged.
Historical reference data, labels, metadata, source hashes and tolerances are
not rewritten for the migration.
Consumer replay validates that this inventory accounts for every originally
recorded paper test file. `--consumer-manifest PATH` can select an explicit
inventory; by default it is resolved beside `--manifest`.

`test_replay_reference.py` exercises controlled defects: a covariance perturbation,
changed asset ordering, missing output, corrupted FAJ prepared inputs, corrupted
MATF-CMA report data, and corrupted Rosaa panel output must fail. The numerical replay also
checks independent covariance reconstruction, QR/asset-space GLS identities and
held-unit basket accounting. No historical code-hash receipt or recorded
tolerance is rewritten to accommodate an API move.
