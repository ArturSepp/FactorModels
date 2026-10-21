"""
Factor-model identity: what a tradable factor model IS, independent of how it is built.

A tradable factor model is a set of factors. Each factor is a long side, an optional short side, and
the volatility the combined basket is held at:

    r_f(t) = l_f(t) · [ Σ_k w^L_fk·r_k(t) − Σ_j w^S_fj·r_j(t) ],    l_f(t) = σ*_f / σ̂_f(t)

with ``w^L`` and ``w^S`` both POSITIVE magnitudes — the short side's sign is in the formula, not in
the numbers — ``σ*_f`` the vol target and ``σ̂_f`` an EWMA estimate of the unlevered basket's
volatility. The leverage ``l_f`` is what makes a regression β against the factor read as an exposure
ratio rather than an artifact of the volatility regime the factor was estimated in.

A ``FactorFamily`` carries what every factor of a kind shares: its vol target, the sign its betas
carry, and whether it is gated. A regional equity factor differs from another only in its basket, so
the family is declared once and inherited. Holding every equity factor at one vol target is the
point, not an approximation: the target is a construction choice, and a common one is what makes
β comparable across regions.

An opt-in ``VolTargetPolicy.NONE`` factor instead fixes construction leverage at one.
Its family still supplies beta rules and the separate CMA volatility initialisation prior.
Legacy declarations retain the inherited volatility target.

Units and conventions:

* vol targets and estimates are ANNUALISED DECIMALS (0.15 = 15% p.a.);
* weights are dimensionless positive magnitudes and are NOT required to sum to one;
* ``vol_span`` is in periods of the construction return frequency (weekly, W-WED), not days;
* ``rebalancing_freq`` is a pandas calendar anchor ('ME', 'QE', 'W-WED').

Construction flattens a factor to one signed weight vector, LONG SIDE FIRST. That ordering is
load-bearing for reproducibility, not for correctness: floating-point addition is not associative,
so reordering a basket moves the built factor in its last bits (measured: 1.7e-13 on a NAV that
starts at 1.0).

Two fields carry semantics the CMA construction reads rather than the panel build:
``regional_equity_factor`` and ``regional_rates_factor`` name the factor each of the two regional
CMA channels keys off. They are None for a model whose regional exposures are factors in their own
right, which switches both channels off without a code change.

Entry points: ``FactorFamily``, ``Factor``, ``FactorModelSpec``, and the memberless ``RiskFactors``
base that concrete factor-set enums subclass.

Not in this module: construction, concrete model catalogues, or vendor access.
Factor transforms reuse ``qis.perfstats.config.ReturnTypes``. The optional
observed-index branch retains raw levels, without a basket NAV or volatility target;
its default-independent return type is explicitly selected.
"""
from __future__ import annotations

# packages
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type
import pandas as pd
from qis.perfstats.config import ReturnTypes


__all__ = [
    "DEFAULT_VOL_WARMUP_PERIOD", "RiskFactors", "FactorFamily", "VolTargetPolicy",
    "Factor", "FactorModelSpec", "assemble_factor_model", "normalise_factor_labels",
]


DEFAULT_VOL_WARMUP_PERIOD: int = 13


class RiskFactors(str, Enum):
    """Memberless base for tradable factor-set enums.

    Concrete factor sets subclass this and define their members. Shared behavior (name listing,
    column validation) lives here so multiple factor sets stay consistent. The qis container
    (``qis.market_data.FactorsData``) accepts any such subclass via its ``factors`` parameter. A
    ``str``-Enum may be subclassed only while it has no members, so this class defines none.

    Members hash equal to their string values, so a dict keyed by members also accepts the value
    strings AND a dict keyed by strings accepts the members — both directions are relied on across
    the stack and neither may be broken.
    """

    @classmethod
    def values(cls, is_upper: bool = False) -> List[str]:
        """the factor names, as strings."""
        if is_upper:
            return [e.value.upper() for e in cls]
        return [e.value for e in cls]

    @classmethod
    def validate_columns(cls, columns: Iterable[str], require_all: bool = True) -> None:
        """Check that ``columns`` cover the factor set; raise if any are missing."""
        missing = [v for v in cls.values() if v not in set(columns)]
        if require_all and missing:
            raise ValueError(f"{cls.__name__}: missing factor columns {missing!r}")


@dataclass(frozen=True)
class FactorFamily:
    """What every factor of one kind shares.

    Declared once and inherited, so adding a regional factor is a basket and a family reference
    rather than a fresh copy of four constants that must not drift.

    The family owns the vol target and both beta rules. A factor may override the vol target — the
    research entry points do, to rebuild a factor at a different target — but not the sign or the
    gate: a factor whose betas obey a different rule is a factor of a different kind.

    Attributes:
        name: Family identifier, e.g. 'Equity'. Recorded in the provenance snapshot.
        vol_target: Annualised decimal vol every factor of this family is held at.
        long_only_sign: Sign constraint the family's betas carry: +1.0 forces a non-negative
            loading for a sign-constrained instrument, -1.0 non-positive, None no explicit sign.
            With automatic signs enabled, None defers to cluster sign selection and its zero gate.
        is_gated: The loading is forced to zero unless the instrument declares the exposure.

    Raises:
        ValueError: If the name is empty, the vol target is not positive, or the sign is not one of
            +1.0 / -1.0 / None.
    """
    name: str
    vol_target: Optional[float]
    long_only_sign: Optional[float] = None
    is_gated: bool = False

    def __post_init__(self) -> None:
        """Validate the family's name, volatility target and loading-sign rule."""
        if not isinstance(self.name, str) or len(self.name) == 0:
            raise ValueError(f"a factor family needs a non-empty name, got {self.name!r}")
        if self.vol_target is not None and not self.vol_target > 0.0:
            raise ValueError(f"{self.name}: vol_target must be a positive annualised decimal, "
                             f"got {self.vol_target!r}")
        if self.long_only_sign not in (1.0, -1.0, None):
            raise ValueError(f"{self.name}: long_only_sign must be +1.0, -1.0 or None, "
                             f"got {self.long_only_sign!r}")


class VolTargetPolicy(str, Enum):
    """Persistent construction policy; legacy None remains family inheritance."""
    INHERIT = 'inherit'
    FIXED = 'fixed'
    NONE = 'none'


@dataclass(frozen=True)
class Factor:
    """One tradable factor: a long side, an optional short side, and how it is held.

    Weights on BOTH sides are positive magnitudes. The short side's sign lives in the flattening,
    not in the declaration, so ``short_side_weights=(-0.5,)`` is a long position by accident and is
    rejected.

    Attributes:
        name: The factor's name, and the column it occupies in a built panel.
        family: The family it inherits its vol target and beta rules from.
        long_side_tickers: Constituents held long, matching columns of the base price panel.
        long_side_weights: Positive magnitudes aligned to ``long_side_tickers``.
        short_side_tickers: Constituents held short, or None for a long-only factor.
        short_side_weights: Positive magnitudes aligned to ``short_side_tickers``, or None.
        vol_target: Overrides the family's vol target for this factor. None inherits.
        vol_target_policy: Explicit NONE disables construction scaling; INHERIT preserves legacy
            resolution, and FIXED requires a numeric override. It never changes family beta rules.
        rebalancing_freq: Pandas calendar anchor the leverage is rebalanced on. A build may
            override it uniformly for the whole model.
        vol_span: EWMA span for the volatility estimate, in periods of the construction return
            frequency (weekly). 3*52 is a ~1.04-year half-life.
        vol_warmup_period: Valid construction returns used for the initial sample covariance;
            exposure starts after this sample is known (default 13 weekly observations).
        require_all_constituents: Start only when every leg is available, rejecting later gaps.
            Used for inseparable hedged trackers; other baskets retain live-leg allocation.
        component_factors: Optional constructed factors matching the basket tickers. Their
            NAVs are built first; the composite retains its own rebalancing frequency even
            when a model-wide override is supplied for its underlying components.
        backfill_factor: Optional separately constructed prehistory. Only dates before this
            factor's first NAV are filled, preserving its newer NAV and holding periods.
        return_type: QIS LOG by default; DIFFERENCE uses changes in native level units.
            The choice applies to estimation, covariance and historical scenarios.
        is_observed_index: Copy a single raw source without a tradable basket, rescaling,
            FX conversion or cash adjustment. Requires NONE volatility-target policy.

    Raises:
        ValueError: If the long side is empty, either side's weights do not align with its tickers,
            a weight is not strictly positive, a constituent appears twice across both sides, only
            one of the short-side fields is given, or the vol target override is not positive.
    """
    name: str
    family: FactorFamily
    long_side_tickers: Tuple[str, ...]
    long_side_weights: Tuple[float, ...]
    short_side_tickers: Optional[Tuple[str, ...]] = None
    short_side_weights: Optional[Tuple[float, ...]] = None
    vol_target: Optional[float] = None
    rebalancing_freq: str = 'ME'
    vol_span: int = 3 * 52
    vol_target_policy: VolTargetPolicy = VolTargetPolicy.INHERIT
    vol_warmup_period: int = DEFAULT_VOL_WARMUP_PERIOD
    require_all_constituents: bool = False
    component_factors: Tuple[Factor, ...] = ()
    backfill_factor: Optional[Factor] = None
    return_type: ReturnTypes = ReturnTypes.LOG
    is_observed_index: bool = False

    def __post_init__(self) -> None:
        """Freeze declared sequences and validate basket, scaling and source rules."""
        if not isinstance(self.return_type, ReturnTypes):
            raise TypeError('return_type must be a qis ReturnTypes member')
        if self.return_type not in (ReturnTypes.LOG, ReturnTypes.DIFFERENCE):
            raise ValueError('Factor estimation supports LOG and DIFFERENCE')
        if self.is_observed_index and (
                len(self.long_side_tickers) != 1 or tuple(self.long_side_weights) != (1.0,)
                or self.short_side_tickers or self.component_factors or self.backfill_factor
                or self.vol_target is not None or self.vol_target_policy is not VolTargetPolicy.NONE):
            raise ValueError('Observed indices require one unit source and no basket/target/backfill')
        # Coerce the sequence fields so a declaration may use a list, and so a constituent may be
        # given as a str-Enum member for type safety while the runtime carries plain strings.
        # str() on a str-Enum member returns the qualified member name in Python
        # 3.11, so the value is taken explicitly. A frozen dataclass needs object.__setattr__.
        object.__setattr__(self, 'long_side_tickers', _as_ticker_tuple(self.long_side_tickers))
        object.__setattr__(self, 'long_side_weights', _as_weight_tuple(self.long_side_weights))
        object.__setattr__(self, 'short_side_tickers',
                           None if self.short_side_tickers is None
                           else _as_ticker_tuple(self.short_side_tickers))
        object.__setattr__(self, 'short_side_weights',
                           None if self.short_side_weights is None
                           else _as_weight_tuple(self.short_side_weights))

        if not isinstance(self.name, str) or len(self.name) == 0:
            raise ValueError(f"a factor needs a non-empty name, got {self.name!r}")
        if not isinstance(self.family, FactorFamily):
            raise ValueError(f"{self.name}: family must be a FactorFamily, got {self.family!r}")
        if len(self.long_side_tickers) == 0:
            raise ValueError(f"{self.name}: the long side needs at least one constituent. A factor "
                             f"that is only short is declared as a long of the inverse.")
        if (self.short_side_tickers is None) != (self.short_side_weights is None):
            raise ValueError(f"{self.name}: give both short_side_tickers and short_side_weights, "
                             f"or neither; got {self.short_side_tickers!r} and "
                             f"{self.short_side_weights!r}")
        for side, tickers, weights in (('long', self.long_side_tickers, self.long_side_weights),
                                       ('short', self.short_side_tickers, self.short_side_weights)):
            if tickers is None:
                continue
            if len(tickers) != len(weights):
                raise ValueError(f"{self.name}: the {side} side's tickers and weights must align, "
                                 f"got {len(tickers)} and {len(weights)}")
            negative = [(t, w) for t, w in zip(tickers, weights) if not w > 0.0]
            if negative:
                raise ValueError(f"{self.name}: {side}-side weights are positive magnitudes — the "
                                 f"short side's sign is applied at construction. Got {negative!r}")
        tickers = self.signed_tickers()
        duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
        if duplicates:
            raise ValueError(f"{self.name}: a constituent may appear once across both sides, "
                             f"got duplicates {duplicates!r}")
        object.__setattr__(self, 'component_factors', tuple(self.component_factors))
        if self.backfill_factor is not None:
            if not isinstance(self.backfill_factor, Factor):
                raise ValueError(f"{self.name}: backfill_factor must be a Factor")
            if self.component_factors:
                raise ValueError(f"{self.name}: a composite cannot also declare a backfill factor")
        if self.component_factors:
            if (not all(isinstance(factor, Factor) for factor in self.component_factors)
                    or tuple(factor.name for factor in self.component_factors) != tickers):
                raise ValueError(f"{self.name}: component factors must match basket tickers in order")
            if self.name in tickers:
                raise ValueError(f"{self.name}: a factor cannot contain itself")
        object.__setattr__(self, 'vol_target_policy', VolTargetPolicy(self.vol_target_policy))
        if self.vol_target_policy is VolTargetPolicy.NONE and self.vol_target is not None:
            raise ValueError('no-target construction cannot also specify a volatility target')
        if self.vol_target_policy is VolTargetPolicy.FIXED and self.vol_target is None:
            raise ValueError('fixed targeting requires a numerical volatility target')
        if self.vol_target is not None and not self.vol_target > 0.0:
            raise ValueError(f"{self.name}: a vol_target override must be positive or None, "
                             f"got {self.vol_target!r}")
        if (isinstance(self.vol_warmup_period, bool)
                or not isinstance(self.vol_warmup_period, Integral)
                or self.vol_warmup_period < 2):
            raise ValueError(f"{self.name}: vol_warmup_period must be an integer of at least 2")
        if self.vol_span < 2:
            raise ValueError(f"{self.name}: vol_span must be at least 2 periods, "
                             f"got {self.vol_span!r}")
        if not isinstance(self.rebalancing_freq, str) or len(self.rebalancing_freq) == 0:
            raise ValueError(f"{self.name}: rebalancing_freq must be a pandas calendar anchor, "
                             f"got {self.rebalancing_freq!r}")

    # ---- resolution against the family -------------------------------------------------

    @property
    def resolved_vol_target(self) -> Optional[float]:
        """Return the construction target, or None for explicitly unscaled exposure."""
        if self.vol_target_policy is VolTargetPolicy.NONE:
            return None
        return self.vol_target if self.vol_target is not None else self.family.vol_target

    @property
    def initial_volatility(self) -> float:
        """Statistical EWMA seed, independent of whether construction targets volatility."""
        return self.vol_target if self.vol_target is not None else self.family.vol_target

    @property
    def long_only_sign(self) -> Optional[float]:
        """the family's beta sign rule. A factor does not override it — see ``FactorFamily``."""
        return self.family.long_only_sign

    @property
    def is_gated(self) -> bool:
        """whether the family's loading is gated behind a per-instrument flag."""
        return self.family.is_gated

    @property
    def is_long_only(self) -> bool:
        """True when the factor has no short side."""
        return self.short_side_tickers is None

    # ---- flattening --------------------------------------------------------------------

    def signed_tickers(self) -> Tuple[str, ...]:
        """every constituent, LONG SIDE FIRST. The order the built basket is summed in."""
        if self.short_side_tickers is None:
            return self.long_side_tickers
        return self.long_side_tickers + self.short_side_tickers

    def signed_weights(self) -> Tuple[float, ...]:
        """the weights aligned to ``signed_tickers``, with the short side negated."""
        if self.short_side_weights is None:
            return self.long_side_weights
        return self.long_side_weights + tuple(-w for w in self.short_side_weights)

    def required_tickers(self) -> Tuple[str, ...]:
        """Leaf price inputs, recursively resolving any constructed component factors."""
        tickers = (tuple(ticker for factor in self.component_factors
                         for ticker in factor.required_tickers())
                   if self.component_factors else self.signed_tickers())
        if self.backfill_factor is not None:
            tickers += self.backfill_factor.required_tickers()
        return tuple(dict.fromkeys(tickers))

    def net_weight(self) -> float:
        """the sum of the signed weights: 0.0 for a funded long-short, 1.0 for a plain long."""
        return sum(self.signed_weights())

    def gross_weight(self) -> float:
        """the sum of the weight magnitudes — the basket's gross exposure before leverage."""
        return sum(abs(w) for w in self.signed_weights())

    def to_text(self) -> str:
        """a one-line, diffable rendering of the whole factor, for a provenance record.

        Complete enough to reconstruct it: a reader comparing two saved cuts sees a changed
        constituent, weight, vol target, anchor, family or sign rule as a changed line. The legs are
        rendered in flattened signed form — what is actually built — so the rendering stays
        comparable with cuts saved before the long/short split.

        Numbers use ``repr``, the shortest string that round-trips to the same float. A fixed
        precision would render 1/3 as 0.333333 and make a genuine weight change indistinguishable
        from rounding — which is the one thing a provenance record must not do.

        Returns:
            str: Constituents, weights, target, cadence, initialisation and family policy.
        """
        legs = ', '.join(f"{t}:{w!r}" for t, w in zip(self.signed_tickers(), self.signed_weights()))
        parts = [legs,
                 f"vol_target={self.resolved_vol_target!r}",
                 self.rebalancing_freq,
                 f"span={self.vol_span}",
                 f"vol_warmup_period={self.vol_warmup_period}",
                 "vol_init=warmup_sample_covariance"]
        if self.long_only_sign is not None:
            parts.append(f"sign={self.long_only_sign:+}")
        if self.is_gated:
            parts.append('gated')
        parts.append(f"family={self.family.name}")
        if self.vol_target_policy is not VolTargetPolicy.INHERIT:
            parts.append(f"vol_target_policy={self.vol_target_policy.value}")
            parts.append(f"initial_volatility={self.initial_volatility!r}")
        if self.require_all_constituents:
            parts.append('require_all_constituents=True')
        if self.component_factors:
            parts.append('components=[' + '; '.join(
                f'{factor.name}: {factor.to_text()}' for factor in self.component_factors) + ']')
        if self.backfill_factor is not None:
            parts.append(f'backfill=[{self.backfill_factor.to_text()}]; splice=preserve_newer_nav')
        if self.return_type is not ReturnTypes.LOG or self.is_observed_index:
            parts.append(f'return_type={self.return_type.name}; observed_index={self.is_observed_index}')
        return ' | '.join(parts)


def _as_ticker_tuple(tickers: Sequence) -> Tuple[str, ...]:
    """constituents as plain strings, accepting str-Enum members in a declaration."""
    return tuple(t.value if isinstance(t, Enum) else str(t) for t in tickers)


def _as_weight_tuple(weights: Sequence) -> Tuple[float, ...]:
    """weights as plain floats."""
    return tuple(float(w) for w in weights)


@dataclass(frozen=True)
class FactorModelSpec:
    """One tradable factor model: its factor set, its factors, and the panel it persists to.

    A model is a value, not a module: one model and a second model with different factors,
    vols or baskets coexist without either being the ambient default. A model is passed as this
    object. ``name`` is stored as provenance; it is not accepted as a runtime model selector.

    Attributes:
        name: Model identifier, recorded in the configuration snapshot of any cut built under it.
        factors: The factor-set enum, a ``RiskFactors`` subclass. Its member order is the column
            order of the built panel.
        baskets: One ``Factor`` per enum member, exactly. Each ``Factor.name`` must equal its
            member's value, so the two declarations cannot drift apart.
        file_name: Panel file this model writes and loads, without extension. Two models must not
            share one.
        regional_equity_factor: The factor the ``(1 - beta)`` regional equity CMA add-on keys off,
            or None to leave that channel dormant. None is correct for a model carrying regional
            equity factors of its own, where the add-on would double count.
        regional_rates_factor: The factor the duration-loaded regional rates CMA bracket keys off,
            or None to leave that channel dormant.
        factor_families: Optional report-family partition of the selected factor identifiers.
            Equal splitting governs family shocks; global construction and beta rules stay intact.
            None preserves the grouping implied by the global construction families.

    Raises:
        ValueError: If the name or file name is empty, the factor set is memberless, a member has
            no factor, a factor has no member, a factor's name disagrees with its member's value,
            or a regional factor is not in the factor set.
    """
    name: str
    factors: Type[RiskFactors]
    baskets: Mapping[RiskFactors, Factor]
    file_name: str
    regional_equity_factor: Optional[RiskFactors] = None
    regional_rates_factor: Optional[RiskFactors] = None
    factor_families: Optional[Mapping[str, Tuple[str, ...]]] = None

    def __post_init__(self) -> None:
        """Validate the factor selection and freeze its basket and group mappings."""
        if not isinstance(self.name, str) or len(self.name) == 0:
            raise ValueError(f"a factor model needs a non-empty name, got {self.name!r}")
        if not isinstance(self.file_name, str) or len(self.file_name) == 0:
            raise ValueError(f"a factor model needs a non-empty file_name, got {self.file_name!r}")
        members = list(self.factors)
        if len(members) == 0:
            raise ValueError(f"{self.factors.__name__} defines no factors")
        missing = [f.value for f in members if f not in self.baskets]
        if missing:
            raise ValueError(f"{self.name}: no factor declared for {missing!r}")
        member_set = set(members)
        extra = [str(k) for k in self.baskets if k not in member_set]
        if extra:
            raise ValueError(f"{self.name}: factors declared for non-members {extra!r}")
        mismatched = [(m.value, self.baskets[m].name) for m in members
                      if self.baskets[m].name != m.value]
        if mismatched:
            raise ValueError(f"{self.name}: a factor's name must equal its enum member's value, "
                             f"got {mismatched!r}")
        for label, factor in (('regional_equity_factor', self.regional_equity_factor),
                              ('regional_rates_factor', self.regional_rates_factor)):
            if factor is not None and factor not in member_set:
                raise ValueError(f"{self.name}: {label}={factor!r} is not in "
                                 f"{self.factors.__name__}")
        # Freeze the mapping: the spec is an immutable snapshot and a mutable dict behind a frozen
        # dataclass is a single source of truth only by convention.
        object.__setattr__(self, 'baskets', MappingProxyType(dict(self.baskets)))
        if self.factor_families is not None:
            resolved = {}
            for name, family_members in self.factor_families.items():
                if not isinstance(name, str) or not name.strip() or name != name.strip():
                    raise ValueError('Factor family names must be nonempty and trimmed')
                if isinstance(family_members, str):
                    raise ValueError('Factor family members must be a sequence, not a string')
                selected = _as_ticker_tuple(family_members)
                if not selected:
                    raise ValueError(f'Factor family {name!r} is empty')
                resolved[name] = selected
            flattened = [item for selected in resolved.values() for item in selected]
            if len(flattened) != len(set(flattened)):
                raise ValueError('Factor families must not overlap or repeat members')
            if set(flattened) != set(self.factor_names()):
                raise ValueError('Factor families must partition all selected factors exactly')
            object.__setattr__(self, 'factor_families', MappingProxyType(resolved))

    def __repr__(self) -> str:
        """a one-line identity. The generated repr inlines every factor, which makes any signature
        or log line carrying a spec as a default argument unreadable."""
        return (f"FactorModelSpec(name={self.name!r}, factors={self.factors.__name__}, "
                f"n_factors={len(self.baskets)}, file_name={self.file_name!r})")

    def basket(self, factor: RiskFactors) -> Factor:
        """the declaration of one factor.

        Args:
            factor: A member of this model's factor set.

        Returns:
            Factor: Its sides, weights, family and holding rule.

        Raises:
            ValueError: If the factor is not in this model.
        """
        if factor not in self.baskets:
            raise ValueError(f"{self.name} has no factor {factor!r}; "
                             f"it carries {self.factor_names()!r}")
        return self.baskets[factor]

    def vol_targets(self) -> Dict[RiskFactors, Optional[float]]:
        """the per-factor annualised vol targets, keyed by enum member.

        Members hash equal to their values, so the returned dict is equally accessible by the
        factor name string. This is the replacement for a module-level ``FACTOR_VOLS`` constant:
        the targets are declared once, on the families, and read from there.

        Returns:
            Dict[RiskFactors, Optional[float]]: Annualised targets; None for explicitly unscaled factors.
        """
        return {member: factor.resolved_vol_target for member, factor in self.baskets.items()}

    def families(self) -> List[FactorFamily]:
        """the distinct families the model draws on, in factor order."""
        seen: Dict[str, FactorFamily] = {}
        for member in self.factors:
            family = self.baskets[member].family
            seen.setdefault(family.name, family)
        return list(seen.values())

    def factors_in_family(self, family: FactorFamily) -> List[RiskFactors]:
        """the members belonging to one family, in factor order."""
        return [m for m in self.factors if self.baskets[m].family.name == family.name]

    def stress_factor_groups(self):
        """Return QIS family definitions with explicit equal total-bump allocation.

        Groups carry the model's declared membership; they never collapse factors
        or covariance. A simple total bump x assigns x/n to each member before
        log1p. Exposure sums and split-bump sensitivities are distinct outputs.
        """
        from qis import FactorGroupSpec
        groups = {}
        for name, members in self.resolved_factor_families().items():
            key = "family:" + name
            if key in self.factor_names():
                raise ValueError(f'Factor group key collides with a factor: {key}')
            groups[key] = FactorGroupSpec(key, members, label=name)
        return groups

    def resolved_factor_families(self) -> Dict[str, Tuple[str, ...]]:
        """Return assembled report families, preserving legacy defaults when omitted."""
        if self.factor_families is not None:
            return dict(self.factor_families)
        return {family.name: tuple(member.value for member in self.factors_in_family(family))
                for family in self.families()}

    def factor_names(self) -> List[str]:
        """the factor names in panel column order."""
        return [factor.value for factor in self.factors]

    def return_types(self) -> Dict[str, ReturnTypes]:
        """Return ordered per-factor transformations; observations retain native units."""
        return {member.value: self.baskets[member].return_type for member in self.factors}

    def required_tickers(self) -> List[str]:
        """every constituent the model needs from the base panel, deduplicated, in use order."""
        seen: Dict[str, None] = {}
        for member in self.factors:
            for ticker in self.baskets[member].required_tickers():
                seen.setdefault(ticker, None)
        return list(seen)


def assemble_factor_model(name: str,
                          factors: Type[RiskFactors],
                          file_name: str,
                          *,
                          catalogue: Mapping[str, Factor],
                          factor_families: Optional[Mapping[str, Tuple[str, ...]]] = None,
                          regional_equity_factor: Optional[RiskFactors] = None,
                          regional_rates_factor: Optional[RiskFactors] = None,
                          ) -> FactorModelSpec:
    """Assemble an immutable model selection from a caller-supplied catalogue.

    The catalogue owns each factor's construction and loading rules. This function
    selects its factors in enum order, with optional reporting groups and regional
    channels. It does not select a default catalogue, access providers or write files.

    Args:
        name: Model identifier recorded as provenance.
        factors: Selected factor enum; each value must identify a catalogue entry.
        file_name: Consumer-managed panel identifier, without a file extension.
        catalogue: Factor definitions keyed by their names, supplied explicitly.
        factor_families: Optional complete partition for downstream family reporting.
        regional_equity_factor: Optional selected factor identifying the equity channel.
        regional_rates_factor: Optional selected factor identifying the rates channel.

    Returns:
        FactorModelSpec: A complete, immutable selection ready for panel construction.

    Raises:
        ValueError: If a selected factor is missing or the resulting specification is invalid.
    """
    missing = [member.value for member in factors if member.value not in catalogue]
    if missing:
        raise ValueError(f"{name}: no catalogue entry for {missing!r}")
    return FactorModelSpec(
        name=name, factors=factors, file_name=file_name,
        baskets={member: catalogue[member.value] for member in factors},
        factor_families=factor_families, regional_equity_factor=regional_equity_factor,
        regional_rates_factor=regional_rates_factor)


def normalise_factor_labels(frame: pd.DataFrame,
                            *,
                            aliases: Mapping[str, str],
                            axis: str = 'columns',
                            ) -> pd.DataFrame:
    """Apply caller-supplied label aliases without changing the numerical payload.

    Args:
        frame: Factor data whose columns or index require canonical names.
        aliases: Explicit mapping from incoming labels to canonical labels.
        axis: Axis to rename: ``'columns'`` or ``'index'``.

    Returns:
        pd.DataFrame: A renamed frame, or the input object if no labels change.

    Raises:
        ValueError: If the axis is invalid or aliases introduce duplicate labels.
    """
    if axis not in ('columns', 'index'):
        raise ValueError("axis must be 'columns' or 'index'")
    labels = frame.columns if axis == 'columns' else frame.index
    renamed = labels.map(lambda label: aliases.get(label, label))
    if labels.equals(renamed):
        return frame
    if renamed.has_duplicates:
        duplicates = renamed[renamed.duplicated()].unique().tolist()
        raise ValueError(f'duplicate factor labels after normalisation: {duplicates!r}')
    return frame.set_axis(renamed, axis=axis)
