"""
Beta sign constraints: what a LASSO fit is allowed to do with each factor loading.

Turns a factor model plus two per-instrument flags into the inequality-constraint matrix the
sign-constrained estimation consumes. Nothing is estimated here — the matrix is a declaration, and
every rule in it is read off the model's baskets rather than written out per factor.

Each factor's FAMILY declares its rule, and every factor of that family inherits it:

* ``long_only_sign = +1.0`` — the loading is forced non-negative for a sign-constrained
  instrument. Other instruments have no explicit sign and inherit automatic sign selection.
  This is the declared long-only economic prior, not an inference from the position quantity.
* ``long_only_sign = -1.0`` — forced non-positive on the same set. The volatility factor: an
  instrument long the risk premia is structurally short volatility.
* ``long_only_sign = None`` — no explicit economic sign, such as inflation, commodities or FX.
  With production automatic signs enabled, these cells inherit the cluster-selected sign and
  t-statistic zero gate. NaN does not disable that estimator layer.
* ``is_gated = True`` — the loading is forced to ZERO unless the instrument declares the exposure,
  and non-negative where it does. This is how a factor that only some instruments may carry stays
  out of everyone else's fit rather than being picked up as noise.

The gating flag implies the sign constraint: an instrument permitted a gated exposure is long the
risk premia by construction, so a caller sets one flag, not two.

Values in the returned frame are the convention the LASSO solver reads:

    +1.0  force the loading non-negative
    -1.0  force it non-positive
     0.0  force it to zero
     NaN  no explicit restriction (automatic signs may still apply)

Flags are declared by the caller for each instrument. They are properties of the
instrument, independent of its descriptive taxonomy. Estimation remains in FactorLasso;
catalogue and universe policy remain with the consumer.

Purpose: Declare economic sign and gating constraints for factor loadings.
Inputs: Factor families, ordered asset identities and per-instrument exposure flags.
Outputs: compute_beta_loading_signs returns asset-by-factor requirements in {-1, 0, +1, NaN}.
Units: Signs and gating flags are dimensionless; NaN means no explicit constraint.
Conditioning: Rules come from declared family metadata; no fitted signs or data-dependent gates are inferred.
Ownership: Analytics declares constraints; estimation and FactorLasso apply them during fitting.
"""
from __future__ import annotations

# packages
import numpy as np
import pandas as pd
# qis / project
from factormodels.analytics.definitions import FactorModelSpec


__all__ = [
    'compute_beta_loading_signs',
]


def compute_beta_loading_signs(spec: FactorModelSpec,
                               long_only_betas: pd.Series,
                               gated_factor_exposure: pd.Series
                               ) -> pd.DataFrame:
    """the sign-constraint matrix for a factor model, from two per-instrument flags.

    Every rule comes off the model's baskets — ``long_only_sign`` and ``is_gated`` — so a model
    with different factors, or the same factors under different economic priors, produces its
    matrix without a line of code changing here.

    A model may declare more than one gated factor, in which case the single
    ``gated_factor_exposure`` flag opens all of them for an instrument. That is deliberate for the
    shared eligibility contract; a model needing per-factor gating wants a flag per factor
    and a signature to match, and should say so rather than reusing this one.

    Args:
        spec: The factor model. Its factor order is the column order of the result.
        long_only_betas: Boolean Series indexed by instrument — the instrument's systematic
            exposures are sign-constrained: long the risk premia and short volatility. The returned
            frame is indexed by this Series. NaN is read as False.
        gated_factor_exposure: Boolean Series indexed by instrument — the instrument may carry a
            non-negative loading on every gated factor, and is sign-constrained on the risk premia
            whether or not ``long_only_betas`` says so. NaN is read as False.

    Returns:
        pd.DataFrame: Indexed by instrument, columns = the model's factors in model order. Values
        +1.0 (non-negative), -1.0 (non-positive), 0.0 (zero), NaN (no explicit restriction;
        inherits automatic signs when the estimator enables them).

    Raises:
        ValueError: If ``gated_factor_exposure`` does not cover every instrument in
            ``long_only_betas``.
    """
    assets = long_only_betas.index
    missing = [a for a in assets if a not in gated_factor_exposure.index]
    if missing:
        raise ValueError(f"gated_factor_exposure does not cover {missing}")
    gated_flags = gated_factor_exposure.reindex(assets).fillna(False).astype(bool)
    # A gated exposure implies the long-only sign pattern, so the caller sets one flag, not two.
    long_only = long_only_betas.fillna(False).astype(bool) | gated_flags

    signs = pd.DataFrame(np.nan, index=assets, columns=spec.factor_names())
    # 1.0 where economic signs apply, NaN where the estimator selects signs automatically.
    # Multiplying by the factor's declared sign gives +1 / -1 without branching per factor.
    mask = pd.Series(np.where(long_only, 1.0, np.nan), index=assets)

    for member in spec.factors:
        factor = spec.basket(member)
        # Both rules resolve through the factor's family, so every factor of a kind is constrained
        # the same way by construction rather than by nine declarations agreeing.
        if factor.is_gated:
            signs[member.value] = 0.0
            signs.loc[gated_flags, member.value] = 1.0
        elif factor.long_only_sign is not None:
            signs[member.value] = factor.long_only_sign * mask
    return signs
