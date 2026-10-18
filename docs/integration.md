# Consume a completed model

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [TradableFactors](https://github.com/ArturSepp/tradablefactors).
Software citation: [CITATION.cff](https://github.com/ArturSepp/tradablefactors/blob/main/CITATION.cff).

The [composed runnable example](https://github.com/ArturSepp/tradablefactors/blob/main/examples/downstream_handoff.py)
fits a synthetic model, supplies covariance and explicitly chosen alphas to OP,
then uses QIS to inspect the accepted portfolio's annual tracking error. It installs
OP explicitly; the TF library has no optimiser dependency.

```console
python -m pip install tradablefactors optimalportfolios
python examples/downstream_handoff.py
```

During the unpublished migration, install the matching local candidate wheels
instead of expecting the public index to contain these APIs.

| Package boundary | Completed object passed onward |
|---|---|
| FL → TF | Fitted statistical state and covariance containers; TF retains diagnostics and conventions |
| TF → OP | Selected annual asset covariance, supplied alphas and optional prepared factor/uncertainty inputs |
| TF → QIS | Existing `qis.RiskModel` from `snapshot.to_risk_model()` |
| OP → QIS | Accepted labelled weights, benchmark, dates and supplied risk model |
| TF → method report | Completed GLS/CMA/covariance result; rendering does not refit |

The example checks solver acceptance, unit exposure, nonnegative weights and a
4% annual tracking-error constraint. Failure remains an explicit solver result,
not a replacement portfolio. Its returned `solution`, `portfolio`, `risk_model`
and scalar `tracking_error` are inspectable after the producer finishes.

Total covariance includes the selected full residual block. Existing QIS native
stress/decomposition limitations remain: a diagonal-reference view must be
explicitly requested and labelled; unsupported native-point stress coordinates
are not silently converted to percent. General stress, backtesting and portfolio
reports belong to QIS. Method-specific factor reports remain in TF and use its
plotting primitives. No new QIS API is introduced here.
