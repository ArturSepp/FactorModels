# Consume a completed model

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

The [composed runnable example](https://github.com/ArturSepp/FactorModels/blob/main/examples/downstream_handoff.py)
fits a synthetic model, supplies covariance and explicitly chosen alphas to OP,
then uses QIS to inspect the accepted portfolio's annual tracking error. It installs
OP explicitly; the FM library has no optimiser dependency.

```console
python -m pip install factormodels "optimalportfolios @ git+https://github.com/ArturSepp/OptimalPortfolios.git@42f38ef9eeb27156b2f8997b701677ed4c1836e4"
python examples/downstream_handoff.py
```

The example uses OP's supplied-input API at the reviewed public revision pinned
above. FactorModels itself installs from PyPI and does not require OP. Run the
example from a FactorModels source checkout.

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
