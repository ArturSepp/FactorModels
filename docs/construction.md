---
myst:
  html_meta:
    description: >-
      Constructing a factormodels factor model: ordered factor NAVs from supplied price histories, explicit basket definitions, volatility targets and loading-sign constraints.
---

# Constructing a factor model

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/FactorModels/commit/b6893a9e37d7cb2353666edf20f416a096ee5943)*

Implemented in [FactorModels](https://github.com/ArturSepp/FactorModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

[FactorModels](https://github.com/ArturSepp/FactorModels) constructs ordered factor NAVs from supplied price histories and explicit basket definitions. The completed construction outputs are the model specification, the NAV/level panel and, when requested, the asset-by-factor loading constraints. Each can be inspected and used downstream without Rosaa. Software citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

## Overview

Define the basket and its volatility policy, then construct its history before estimating
asset exposures. The result is a reusable factor panel. QIS implements returns, covariance
recursions and portfolio simulation; FM supplies the construction order and declarations.

<a id="inputs-and-outputs"></a>

## Inputs, notation, and assumptions

| Object | Required information | Completed output and downstream use |
|---|---|---|
| `FactorFamily` and `Factor` | Named price columns, positive long/short magnitudes, volatility policy, rebalance rule and optional components/prehistory | Explicit definitions and signed basket ordering, usable for construction and provenance |
| `assemble_factor_model` | Ordered `RiskFactors` selection and an explicit catalogue | Immutable `FactorModelSpec` with factor order, source requirements, return types and family/sign rules |
| `build_factor_panel` | Specification, supplied price panel and explicit construction overrides | Ordered date-by-factor NAV/level `DataFrame` for estimation, inspection and scenario inputs |
| `compute_beta_loading_signs` | Specification and per-asset sign/eligibility flags | Asset-by-factor `DataFrame` supplied to the estimator; it does not estimate loadings |

The construction panel is a complete result of this step. It does not contain fitted asset loadings or estimated asset covariance. `FactorCovarEstimator` produces the fitted model directly from supplied histories; input persistence belongs to the caller. Retain the specification and actual construction overrides beside a saved panel because a plain `DataFrame` does not record them automatically.

<a id="construction-and-timing"></a>

## Methodology

The implementation uses [QIS](https://github.com/ArturSepp/QuantInvestStrats) for return preparation and portfolio backtesting; see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff). Long legs precede short legs in the signed weight vector. Magnitudes need not sum to one. The default construction observations are weekly Wednesday log returns, with annualised decimal volatility targets and an EWMA span measured in observations.

Volatility-targeted construction starts from the sample covariance of the first 13 valid return observations by default. It takes no exposure before that warm-up and updates the covariance only with subsequent observations. A weight decided at a date applies to the following interval. QIS holds units between rebalance dates, so a drifting basket is not a repeated fixed-weight average of returns.

### Basket coordinates and live constituents

Let $P_{i,t}$ be a supplied constituent price, $w_i$ its signed strategic weight and
$r_{i,t}=\log(P_{i,t}/P_{i,t-1})$ its sampled log return. The basket return used to
understand the risk target is $r_{f,t}=\sum_i w_i r_{i,t}$ for a fixed, fully live basket.
This linear log-return expression describes the risk estimate; the simulated portfolio
NAV follows QIS's held units and traded prices.

Missing initial histories are handled separately on the long and short sides. If $W_s$
is the declared absolute weight of side $s$, and $L_s(t)$ its live constituents, the
decision weight is

$$
\widetilde w_{i,t}=w_i\frac{W_s}{\sum_{j\in L_s(t)}\lvert w_j\rvert},
\qquad i\in L_s(t).
$$

An absent constituent has zero weight; an entirely absent side has zero exposure. With
all constituents present the scale is one. Normalizing long and short sides separately
also covers a zero-net basket, whose row sum cannot be used as a divisor.

### Causal volatility and the decision calendar

For a fully observed, fixed basket, the first $m$ valid observations initialize the
sample covariance $C_0$ with denominator $m-1$. With annualization factor $A$, span $s$
and $\lambda=1-2/(s+1)$, subsequent QIS updates use

$$
C_t=\lambda C_{t-1}+(1-\lambda)r_t r_t^{\mathsf T},\qquad
\widehat\sigma_{f,t}=\sqrt{A\,\widetilde w_{t-1}^{\mathsf T}C_t\widetilde w_{t-1}}.
$$

The initialization observations enter once. FM checks valid observations and held-leg
availability before forming the seed. Unheld missing legs contribute zero; a missing
held return cannot qualify an observation for initialization.

For target $\sigma_f^*$, decision leverage is $\ell_t=\sigma_f^*/\widehat\sigma_{f,t}$
and target holdings are $\ell_t\widetilde w_t$. The QIS finite reciprocal gives zero
exposure while volatility is unavailable. At a rebalance label FM takes the last risk
weight in that period. With weekly Wednesday observations this uses the last available
Wednesday estimate at or before the period end.

QIS maps a non-trading rebalance label to the next observed price date and holds units
between decisions. A label beyond the last supplied price is omitted from the returned
weights. For example, a 31 October 2026 month-end label can execute on 2 November if that
price is available; it cannot execute in a history ending on 30 October. Volatility
estimation cadence, decision cadence and the execution price grid are distinct.

Volatility targeting is configured on `Factor`, before calling `compute_factor_nav`. `VolTargetPolicy.INHERIT` with `vol_target=None` uses the family's target; an existing numeric factor override remains supported. `VolTargetPolicy.NONE` with `vol_target=None` disables targeting, and `VolTargetPolicy.FIXED` with a positive `vol_target` selects an explicit target. Use `dataclasses.replace(factor, vol_target_policy=VolTargetPolicy.FIXED, vol_target=0.10)` for a separate 10% research configuration without changing the original factor. There is no call-time `vol_target` argument or custom inheritance sentinel. `rebalancing_freq=None` and `vol_span=None` in the builder preserve each basket's settings. A consumer can pass a uniform rebalance override explicitly. Composite factors build their components first and retain their own rebalance rule. Supported prehistory splicing scales the older independently constructed NAV to the new series at the join and preserves the newer NAV unchanged.

If a factor inherits from a family whose target is None, it is unscaled. Declaring
`VolTargetPolicy.NONE` makes that choice explicit without changing the existing inheritance rule.

### Composites, backfill and observed levels

A composite constructs each component NAV first, then combines those NAVs using the
composite's own rebalance rule. A model-wide component override does not silently turn
a daily composite into a quarterly drifting portfolio.

At a backfill join $t_0$, the older independently constructed NAV $V^{old}$ is scaled by
$V^{new}_{t_0}/V^{old}_{t_0}$. Only observations before $t_0$ are replaced; every newer
NAV level is preserved. The older history must be complete, finite and positive through
the join. This splice is a historical construction choice supplied by the caller.

An observed-index factor copies native source levels instead of constructing a tradable basket. Its declared return type distinguishes log returns from native-unit differences. Mixed NAV/level panels therefore require the specification's `return_types()` when estimating or interpreting risk; positive raw levels do not imply a normalised tradable return series.

A basket with `require_all_constituents=True` starts when every leg has a price and
raises on a later gap. Other baskets hold the live legs and scale each side to its
declared exposure. Source acquisition and any economic proxy choice belong to the caller.

<a id="offline-example"></a>

## Worked example

The [canonical runnable example](https://github.com/ArturSepp/FactorModels/blob/main/examples/construct_factors.py) defines two synthetic baskets, supplies 520 business-day observations, constructs their panel and checks the loading constraints. It needs only FM and its declared dependencies, with no provider credentials or private catalogue.

```console
python examples/construct_factors.py
```

The example returns `(spec, panel, signs)`. Its gated exposure flag is deliberately shared across all gated factors: eligibility opens each gated loading and also activates the non-gated economic sign pattern. The resulting values are `+1` for nonnegative, `-1` for nonpositive, `0` for fixed zero, and `NaN` for no explicit sign. `NaN` does not disable an estimator's automatic sign policy.

A one-asset basket with annual estimated volatility 20% and target 10% has leverage
0.5 and a target asset weight of 0.5. For a long-short declaration `(1, -1)`, both live
sides retain their own magnitudes; their zero net weight does not imply zero risk.
The warm-up count is in valid return observations, so its elapsed calendar duration also
depends on missing data and the first subsequent rebalance. Thirteen observations need
not mean thirteen weeks of flat NAV in a monthly-rebalanced daily price history.

## Implementation in factormodels

Use `FactorFamily`, `Factor` and `assemble_factor_model` to declare construction, then
`build_factor_panel` or `compute_factor_nav` to obtain histories. Use
`compute_volatility_targeted_portfolio` to inspect decision weights. The
[Python reference](api_reference.md) provides parameter signatures; the
[canonical script](https://github.com/ArturSepp/FactorModels/blob/main/examples/construct_factors.py)
contains the complete synthetic inputs and assertions. Preserve the actual cadence/span
overrides alongside saved panels for reproduction.

The source-adjacent construction tests independently check sample initialization,
the first EWMA update, held-unit accounting, rebalance timing, prehistory splicing and
invariance to changed future prices. The period-label tests include month, quarter and
weekly schedules and a weekend period end with an available following trading date.

<a id="consumer-boundary"></a>

## Interpretation and limitations

Catalogues and aliases are supplied explicitly. `normalise_factor_labels(frame, aliases=...)` applies a caller-owned mapping and rejects resulting collisions. FM contains no production catalogue, provider client, private alias map or scheduling policy. Rosaa supplies those choices through its own assembly and data boundaries.

The subsequent flow is construction → factor estimation using FactorLasso → a completed FM fitted model → supplied covariance/alphas for optimisation or a QIS risk view for risk/stress reporting. See the [canonical API](api.md) for the fitted-result contract and the [migration guide](migration.md) for the direct import changes.

Target volatility is an ex-ante estimate, not a realized-volatility guarantee. A short
seed, changing correlations and a slower decision cadence can all produce different
realized risk. Keep price currency, financing and total/excess-return conventions
consistent before constructing a basket. Observed native-unit levels are distinct from
tradable NAVs even when both appear in one model panel.

## See also

- [Completed model fields and units](completed_model.md).
- [Factor covariance estimation](factor_covariance_hcgl.md).
- [Reference workflows and local paths](reference_workflows.md).

## References

- [FM construction implementation](https://github.com/ArturSepp/FactorModels/blob/main/src/factormodels/analytics/construction.py).
- [FM construction definitions](https://github.com/ArturSepp/FactorModels/blob/main/src/factormodels/analytics/definitions.py).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorModels software citation](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).
