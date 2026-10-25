---
myst:
  html_meta:
    description: >-
      Consume a completed factormodels model downstream: export to a qis RiskModel and hand the fitted covariance to optimalportfolios without refitting.
---

# Consume a completed model

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/91f424eeabc1cd99880aef829feb6af9c61b046b)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

The [composed runnable example](https://github.com/ArturSepp/FactorModels/blob/main/examples/downstream_handoff.py)
fits a synthetic model, supplies covariance and explicitly chosen alphas to OP,
then uses QIS to inspect the accepted portfolio's annual tracking error. Install
OP explicitly; the FM library has no optimiser dependency.

```console
python examples/downstream_handoff.py
```

This development example requires matching candidate installations of FM and OP:
it imports `OpportunityModel` and `solve_opportunity` from
`optimalportfolios.optimization.opportunity`. The old `optimalportfolios.matf`
package has been removed from the OP development source. Build and install candidate
wheels using each repository's contributor instructions; the published OP release
does not yet expose this path. Run the example from the matching FactorModels source
checkout. No runtime fallback to the historical module is supplied.

| Package boundary | Completed object passed onward |
|---|---|
| FL → FM | Fitted statistical state and covariance containers; FM retains diagnostics and conventions |
| FM → OP | Selected annual asset covariance, supplied alphas and optional prepared factor/uncertainty inputs |
| FM → QIS | Existing `qis.RiskModel` from `snapshot.to_risk_model()` |
| OP → QIS | Accepted labelled weights, benchmark, dates and supplied risk model |
| FM → method report | Completed GLS/CMA/covariance result; rendering does not refit |

The example checks solver acceptance, unit exposure, nonnegative weights and a
4% annual tracking-error constraint. Failure remains an explicit solver result,
not a replacement portfolio. Its returned `solution`, `portfolio`, `risk_model`
and scalar `tracking_error` are inspectable after the producer finishes.

Total covariance includes the selected full residual block. Existing QIS native
stress/decomposition limitations remain: a diagonal-reference view must be
explicitly requested and labelled; unsupported native-point stress coordinates
are not silently converted to percent. General stress, backtesting and portfolio
reports belong to QIS. Method-specific factor reports remain in FM and use its
plotting primitives. No new QIS API is introduced here.
