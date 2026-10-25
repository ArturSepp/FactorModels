## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results. For numerical changes, name the independent
reference-value, parity, or finite-difference cross-check.

## Checklist

- [ ] Tests cover the changed behaviour or defect.
- [ ] Return types, cadences, units, cutoffs and provenance conventions remain explicit.
- [ ] Numerical results were checked independently; expected values were not changed to fit output.
- [ ] Every fit stays causal at its cutoff; no history after the cutoff enters an estimate.
- [ ] No optimiser or other consumer dependency was added; statistical estimation stays in `factorlasso` and risk export in `qis`.
- [ ] No proprietary data, local paths, generated output, or development environments are included.
- [ ] `uv run --no-sync pytest` and the relevant lint/docs/wheel checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] Public-signature, default, dependency-floor, or runtime-dependency changes are called out explicitly.
