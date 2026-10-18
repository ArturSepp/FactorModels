# Numerical migration reference

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

Run historical references in the preserved environment against the frozen old
source. Run TF/consumer checks in an explicit composed environment if checking
OP optimisation and private Rosaa construction as well as TF. These are
integration checks; OP and Rosaa are not TF core dependencies. Keep generated
state, environments and numerical outputs outside the source checkout.

`tradablefactors` mode deliberately holds the unchanged downstream reference
consumers fixed while checking the extracted estimator/analytics. `consumer`
mode uses the OP and Rosaa modules resolved by the selected installed consumer
environment (or its explicit source paths), so a later migration cannot pass by
silently replaying the frozen Rosaa implementation.

`test_replay_reference.py` exercises controlled defects: a covariance perturbation,
changed asset ordering and missing output must fail. The numerical replay also
checks independent covariance reconstruction, QR/asset-space GLS identities and
held-unit basket accounting. No historical code-hash receipt or recorded
tolerance is rewritten to accommodate an API move.
