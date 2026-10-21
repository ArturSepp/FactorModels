# Constructing a factor model

*Author: [Artur Sepp](https://github.com/ArturSepp)*

[FactorModels](https://github.com/ArturSepp/FactorModels) constructs ordered factor NAVs from supplied price histories and explicit basket definitions. The completed construction outputs are the model specification, the NAV/level panel and, when requested, the asset-by-factor loading constraints. Each can be inspected and used downstream without Rosaa. Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

## Inputs and outputs

| Object | Required information | Completed output and downstream use |
|---|---|---|
| `FactorFamily` and `Factor` | Named price columns, positive long/short magnitudes, volatility policy, rebalance rule and optional components/prehistory | Explicit definitions and signed basket ordering, usable for construction and provenance |
| `assemble_factor_model` | Ordered `RiskFactors` selection and an explicit catalogue | Immutable `FactorModelSpec` with factor order, source requirements, return types and family/sign rules |
| `build_factor_panel` | Specification, supplied price panel and explicit construction overrides | Ordered date-by-factor NAV/level `DataFrame` for estimation, inspection and scenario inputs |
| `compute_beta_loading_signs` | Specification and per-asset sign/eligibility flags | Asset-by-factor `DataFrame` supplied to the estimator; it does not estimate loadings |

The construction panel is a complete result of this step. It does not contain fitted asset loadings, estimated asset covariance or a portable bundle manifest. Those are separate outputs: `FactorCovarEstimator` produces the fitted model; `save_bundle` persists explicitly assembled bundle inputs separately. Retain the specification and actual construction overrides beside a saved panel because a plain `DataFrame` does not record them automatically.

## Construction and timing

The implementation uses [QIS](https://github.com/ArturSepp/QuantInvestStrats) for return preparation and portfolio backtesting; see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff). Long legs precede short legs in the signed weight vector. Magnitudes need not sum to one. The default construction observations are weekly Wednesday log returns, with annualised decimal volatility targets and an EWMA span measured in observations.

Volatility-targeted construction starts from the sample covariance of the first 13 valid return observations by default. It takes no exposure before that warm-up and updates the covariance only with subsequent observations. A weight decided at a date applies to the following interval. QIS holds units between rebalance dates, so a drifting basket is not a repeated fixed-weight average of returns.

`VolTargetPolicy.NONE` disables targeting explicitly; a missing per-factor target inherits the family's target. `rebalancing_freq=None` and `vol_span=None` in the builder preserve each basket's settings. A consumer can pass a uniform rebalance override explicitly. Composite factors build their components first and retain their own rebalance rule. Supported prehistory splicing scales the older independently constructed NAV to the new series at the join and preserves the newer NAV unchanged.

An observed-index factor copies native source levels instead of constructing a tradable basket. Its declared return type distinguishes log returns from native-unit differences. Mixed NAV/level panels therefore require the specification's `return_types()` when estimating or interpreting risk; positive raw levels do not imply a normalised tradable return series.

Missing-data behaviour is preserved from the extracted construction: jointly required baskets start at joint inception and reject later gaps; other baskets retain the established live-leg allocation. This interface does not fetch, replace or fabricate missing histories.

## Offline example

The [canonical runnable example](https://github.com/ArturSepp/FactorModels/blob/main/examples/construct_factors.py) defines two synthetic baskets, supplies 520 business-day observations, constructs their panel and checks the loading constraints. It needs only FM and its declared dependencies, with no provider credentials or private catalogue.

```console
python examples/construct_factors.py
```

The example returns `(spec, panel, signs)`. Its gated exposure flag is deliberately shared across all gated factors: eligibility opens each gated loading and also activates the non-gated economic sign pattern. The resulting values are `+1` for nonnegative, `-1` for nonpositive, `0` for fixed zero, and `NaN` for no explicit sign. `NaN` does not disable an estimator's automatic sign policy.

## Consumer boundary

Catalogues and aliases are supplied explicitly. `normalise_factor_labels(frame, aliases=...)` applies a caller-owned mapping and rejects resulting collisions. FM contains no production catalogue, provider client, private alias map or scheduling policy. Rosaa supplies those choices through its own assembly and data boundaries.

The subsequent flow is construction → factor estimation using FactorLasso → a completed FM fitted model → supplied covariance/alphas for optimisation or a QIS risk view for risk/stress reporting. See the [canonical API](api.md) for the fitted-result contract and the [migration guide](migration.md) for the direct import changes.
