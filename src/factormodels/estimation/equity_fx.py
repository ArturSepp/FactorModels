"""Point-in-time equity/FX covariance augmentation of a completed local model.

Purpose: Add structural currency risk and retain estimated residual/FX dependence.
Inputs: A local log-excess model, native factor histories, arithmetic forward payoffs and dated exposures.
Outputs: Owned completed joint moments, full/ablated asset covariance and QIS risk interchange.
Units: Annual covariance; local responses are log excess, FX factors are arithmetic notional increments.
Conditioning: One trailing complete-case grid; explicit preservation/re-estimation and PSD policies.
Ownership: FM assembles augmented models; QIS estimates moments and consumes portfolio risk views.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Mapping
import warnings

import numpy as np
import pandas as pd
import qis

from factormodels.analytics.models import FactorModelSnapshot

__all__ = ['EquityFxRiskSpec', 'EquityFxRiskSnapshot', 'build_currency_exposure',
           'augment_equity_fx_model']


@dataclass(frozen=True)
class EquityFxRiskSpec:
    """Explicit common-sample estimation and covariance-preservation policy.

    Args:
        frequency: Observation/reset cadence, never inferred from missing data.
        covariance_policy: Preserve supplied F/R/Sigma or jointly re-estimate all moments.
        window: Maximum number of grid rows before complete-case filtering.
        min_observations: Required complete observations; no fill or pairwise estimation.
        estimator: Sample covariance or the QIS EWMA recursion.
        span: EWMA span in observations, used only by the EWMA estimator.
        mean_adjustment: Sample centring, causal EWMA centring, or no centring for EWMA.
        cross_covariance_policy: Reject incompatible preserved blocks, or explicitly scale
            both FX cross blocks together to the PSD boundary; local F/R and FX variance stay fixed.

    Local responses must be log excess. Arithmetic FX forwards enter as a first-order
    approximation to their reference log-excess contribution, not as logged forward NAVs.
    The specification does not declare that the approximation is empirically adequate.
    """
    frequency: str
    covariance_policy: Literal['preserve_local', 'joint_reestimate']
    window: int = 60
    min_observations: int = 36
    estimator: Literal['sample', 'ewma'] = 'sample'
    span: int = 36
    mean_adjustment: Literal['sample', 'ewma', 'none'] = 'sample'
    cross_covariance_policy: Literal['raise', 'shrink_to_psd'] = 'raise'

    def __post_init__(self) -> None:
        """Reject ambiguous units, estimation rules and unreviewed repair policies."""
        if self.covariance_policy not in ('preserve_local', 'joint_reestimate'):
            raise ValueError('Choose preserve_local or joint_reestimate explicitly')
        if self.cross_covariance_policy not in ('raise', 'shrink_to_psd'):
            raise ValueError('Unknown cross covariance policy')
        if self.covariance_policy == 'joint_reestimate' and self.cross_covariance_policy != 'raise':
            raise ValueError('Joint re-estimation requires no cross-block repair policy')
        for name in ('window', 'min_observations', 'span'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        if self.min_observations > self.window:
            raise ValueError('min_observations exceeds window')
        if ((self.estimator == 'sample' and self.mean_adjustment != 'sample')
                or (self.estimator == 'ewma' and self.mean_adjustment not in ('ewma', 'none'))
                or self.estimator not in ('sample', 'ewma')):
            raise ValueError('Sample covariance uses sample centring; EWMA uses ewma or none')
        offset = pd.tseries.frequencies.to_offset(self.frequency)
        if offset is None or offset.n <= 0:
            raise ValueError('frequency must be a positive observation cadence')
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            try:
                annualization = qis.get_annualization_factor(self.frequency)
            except UserWarning as exc:
                raise ValueError('frequency must have a known annualisation factor') from exc
        if not np.isfinite(annualization) or annualization <= 0:
            raise ValueError('frequency must have a positive annualisation factor')


def _dated(panel: pd.DataFrame, role: str) -> None:
    """Require an ordered, uniquely labelled date panel without infinite observations."""
    if (not isinstance(panel, pd.DataFrame) or panel.empty
            or not isinstance(panel.index, pd.DatetimeIndex) or panel.index.hasnans
            or not panel.index.is_unique or not panel.index.is_monotonic_increasing
            or not panel.columns.is_unique or panel.columns.hasnans):
        raise ValueError(f'{role} requires unique ordered dates and column labels')


def _metadata(value: pd.Series | pd.DataFrame | float, cutoff: pd.Timestamp,
              assets: pd.Index | None, role: str) -> pd.Series:
    """Select metadata available at a decision boundary, never a future row."""
    if isinstance(value, pd.DataFrame):
        _dated(value, role)
        past = value.loc[:cutoff].ffill()
        if past.empty:
            raise ValueError(f'{role} is unavailable at cutoff')
        result = past.iloc[-1]
    elif isinstance(value, pd.Series):
        result = value.copy()
    elif assets is not None and np.isscalar(value):
        result = pd.Series(value, index=assets)
    else:
        raise TypeError(f'{role} requires a labelled Series or dated DataFrame')
    if not result.index.is_unique or result.isna().any():
        raise ValueError(f'{role} must cover every asset exactly once at cutoff')
    if assets is not None:
        if set(assets) != set(result.index):
            raise ValueError(f'{role} labels must match assets exactly')
        result = result.reindex(assets)
    return result


def build_currency_exposure(
    local_ccys: pd.Series | pd.DataFrame, hedge_ratios: float | pd.Series | pd.DataFrame,
    currencies: list[str] | pd.Index, *, reference_ccy: str, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Build opening structural exposures H[i,c]=1-h[i] for the next holding interval.

    Args:
        local_ccys: Static quote-currency IDs or dated currency spells, not domicile.
        hedge_ratios: Additional principal hedges; embedded hedges remain in supplied prices.
        currencies: Exact ordered FX factor IDs, excluding the reference currency.
        reference_ccy: Common reporting currency; its direct structural exposure is zero.
        cutoff: Decision date; dated metadata is selected as-of, with no future filling.

    Currency subunits/ADRs need explicit metadata. No alias or rebasing conversion is guessed.
    Publication availability is separately declared to augment_equity_fx_model.
    """
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or not isinstance(reference_ccy, str) or not reference_ccy.strip():
        raise ValueError('Provide cutoff and a nonblank reference currency')
    ids = pd.Index(currencies)
    if (ids.empty or not ids.is_unique or reference_ccy in ids
            or any(not isinstance(c, str) or not c.strip() for c in ids)):
        raise ValueError('FX currencies must be unique nonblank non-reference IDs')
    ccys = _metadata(local_ccys, cutoff, None, 'local_ccys')
    if any(not isinstance(c, str) or not c.strip() for c in ccys):
        raise ValueError('local_ccys must contain nonblank currency IDs')
    unknown = set(ccys) - set(ids) - {reference_ccy}
    if unknown:
        raise ValueError(f'Missing FX currency factors: {sorted(unknown)}')
    hedges = _metadata(hedge_ratios, cutoff, ccys.index, 'hedge_ratios').astype(float)
    if not np.isfinite(hedges.to_numpy()).all():
        raise ValueError('hedge_ratios must be finite')
    return pd.DataFrame({ccy: ccys.eq(ccy).astype(float) * (1.0 - hedges) for ccy in ids})


def _psd(matrix: np.ndarray, role: str) -> float:
    """Validate finite symmetric PSD moments using a scale-aware roundoff tolerance."""
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14):
        raise ValueError(f'{role} must be finite and symmetric')
    smallest = float(np.linalg.eigvalsh(matrix).min())
    tolerance = 1e-10 * max(float(np.max(np.abs(matrix))), 1e-12)
    if smallest < -tolerance:
        raise ValueError(f'{role} is not positive semidefinite (minimum eigenvalue {smallest:g})')
    return smallest


# QIS RiskModel consumes covariance; it does not reconcile cross blocks against preserved FM marginals.
def _compatible_cross(a: np.ndarray, c: np.ndarray, v: np.ndarray,
                      policy: str) -> tuple[np.ndarray, float]:
    """Reject, or explicitly scale, cross covariance while preserving both PSD marginals."""
    def inverse_root(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the inverse square root and support projector without adding variance."""
        vals, vecs = np.linalg.eigh(matrix)
        keep = vals > 1e-12 * max(float(vals.max()), 1e-12)
        basis = vecs[:, keep]
        return (basis / np.sqrt(vals[keep])) @ basis.T, basis @ basis.T
    ai, ap = inverse_root(a)
    vi, vp = inverse_root(v)
    if not np.allclose(c, ap @ c @ vp, rtol=1e-8, atol=1e-12):
        raise ValueError('FX cross covariance lies outside the preserved marginal support')
    norm = float(np.linalg.svd(ai @ c @ vi, compute_uv=False).max())
    if norm <= 1.0 + 1e-10:
        return c.copy(), 1.0
    if policy == 'raise':
        raise ValueError('Incompatible preserved joint covariance; choose explicit shrink_to_psd '
                         'or joint_reestimate and review the recorded change')
    scale = (1.0 - 1e-12) / norm
    return c * scale, scale


def _moments(panel: pd.DataFrame, spec: EquityFxRiskSpec) -> pd.DataFrame:
    """Delegate common-sample covariance to QIS and annualise exactly once."""
    if spec.estimator == 'sample':
        result = qis.compute_masked_covar_corr(panel, is_covar=True, bias=False)
    else:
        data = panel
        if spec.mean_adjustment == 'ewma':
            data = (data - qis.compute_ewm(data, span=spec.span)).iloc[1:]
        result = pd.DataFrame(qis.compute_ewm_covar(data.to_numpy(), span=spec.span),
                              index=panel.columns, columns=panel.columns)
    return result * qis.get_annualization_factor(spec.frequency)


@dataclass(frozen=True, init=False)
class EquityFxRiskSnapshot:
    """Completed augmented moments with owned data and explicit approximation provenance.

    Covariance is authoritative; the QIS view deliberately omits a diagonal-residual
    factor decomposition, because cross-factor/residual dependence cannot be represented by it.
    Full joint moments, estimated residual/FX dependence and supplied local policy remain inspectable.
    """
    cutoff: pd.Timestamp
    reference_ccy: str
    _fields: dict

    def __init__(self, *, cutoff: pd.Timestamp, reference_ccy: str,
                 covariance: pd.DataFrame, without_residual_fx: pd.DataFrame,
                 local_covariance: pd.DataFrame, joint_covariance: pd.DataFrame,
                 raw_joint_covariance: pd.DataFrame, loadings: pd.DataFrame,
                 residual_fx_covariance: pd.DataFrame, factor_fx_covariance: pd.DataFrame,
                 residual_covariance: pd.DataFrame, sample_index: pd.DatetimeIndex,
                 factor_return_types: dict, provenance: dict, diagnostics: dict) -> None:
        """Own all completed fields; public retrieval never exposes mutable storage."""
        _psd(covariance.to_numpy(), 'augmented asset covariance')
        _psd(joint_covariance.to_numpy(), 'augmented joint covariance')
        fields = dict(covariance=covariance, without_residual_fx=without_residual_fx,
                      local_covariance=local_covariance, joint_covariance=joint_covariance,
                      raw_joint_covariance=raw_joint_covariance, loadings=loadings,
                      residual_fx_covariance=residual_fx_covariance,
                      factor_fx_covariance=factor_fx_covariance, residual_covariance=residual_covariance,
                      sample_index=sample_index, factor_return_types=factor_return_types,
                      provenance=provenance, diagnostics=diagnostics)
        object.__setattr__(self, '_fields', deepcopy(fields))
        object.__setattr__(self, 'cutoff', pd.Timestamp(cutoff))
        object.__setattr__(self, 'reference_ccy', reference_ccy)

    @property
    def covariance(self) -> pd.DataFrame:
        """Return the authoritative annual asset covariance including residual/FX dependence."""
        return deepcopy(self._fields['covariance'])

    @property
    def without_residual_fx(self) -> pd.DataFrame:
        """Return the labelled omission diagnostic; it need not be a valid joint covariance."""
        return deepcopy(self._fields['without_residual_fx'])

    @property
    def local_covariance(self) -> pd.DataFrame:
        """Return the local asset block actually used under the declared policy."""
        return deepcopy(self._fields['local_covariance'])

    @property
    def joint_covariance(self) -> pd.DataFrame:
        """Return selected annual covariance of local factors, residuals and FX payoffs."""
        return deepcopy(self._fields['joint_covariance'])

    @property
    def raw_joint_covariance(self) -> pd.DataFrame:
        """Return unmodified common-sample moments before any marginal preservation."""
        return deepcopy(self._fields['raw_joint_covariance'])

    @property
    def loadings(self) -> pd.DataFrame:
        """Return local fitted B plus structural H; FX dependence of residuals is separate."""
        return deepcopy(self._fields['loadings'])

    @property
    def residual_fx_covariance(self) -> pd.DataFrame:
        """Return the selected annual residual/FX cross covariance, not a diagonal proxy."""
        return deepcopy(self._fields['residual_fx_covariance'])

    @property
    def factor_fx_covariance(self) -> pd.DataFrame:
        """Return the selected annual local-factor/FX cross covariance."""
        return deepcopy(self._fields['factor_fx_covariance'])

    @property
    def residual_covariance(self) -> pd.DataFrame:
        """Return the complete selected residual block, including off-diagonal entries."""
        return deepcopy(self._fields['residual_covariance'])

    @property
    def sample_index(self) -> pd.DatetimeIndex:
        """Return the exact shared observations used for every estimated block."""
        return self._fields['sample_index'].copy()

    @property
    def provenance(self) -> dict:
        """Return coordinates, cutoffs, estimator, original policy and availability declarations."""
        return deepcopy(self._fields['provenance'])

    @property
    def diagnostics(self) -> dict:
        """Return explicit changes, missing-data counts and cross-covariance scale."""
        return deepcopy(self._fields['diagnostics'])

    def to_risk_model(self, *, include_residual_fx: bool = True) -> qis.RiskModel:
        """Export QIS covariance risk and frozen loadings without a false F/diagonal-D split.

        Args:
            include_residual_fx: False exports the omission diagnostic only if its asset
                covariance is PSD; it never repairs an invalid ablation silently.
        """
        covariance = self.covariance if include_residual_fx else self.without_residual_fx
        _psd(covariance.to_numpy(), 'selected diagnostic covariance')
        return qis.RiskModel(covar={self.cutoff: covariance},
                             factor_loadings={self.cutoff: self.loadings},
                             factor_return_types=deepcopy(self._fields['factor_return_types']))


# build_risk_model exports completed covariance; it cannot estimate equity/FX or residual/FX cross blocks.
def augment_equity_fx_model(
    local_model: FactorModelSnapshot, factor_returns: pd.DataFrame,
    local_log_excess_returns: pd.DataFrame, fx_forward_payoffs: pd.DataFrame,
    structural_exposures: pd.DataFrame, *, spec: EquityFxRiskSpec,
    factor_return_types: Mapping, reference_ccy: str,
    metadata_available_date: pd.Timestamp, data_available_date: pd.Timestamp,
) -> EquityFxRiskSnapshot:
    """Augment a local fitted model with first-order structural reference-currency risk.

    Args:
        local_model: Completed local log-excess fit; cutoff determines every history slice.
        factor_returns: Native LOG/DIFFERENCE increments, not reconstructed FX NAVs.
        local_log_excess_returns: Own-currency log asset growth minus log cash growth.
        fx_forward_payoffs: SIMPLE long-local forward P&L per opening reference notional.
        structural_exposures: Asset-by-currency H, fixed at the decision cutoff for the next interval.
        spec: Explicit common-sample moment, original-block and incompatibility policies.
        factor_return_types: Exact native factor coordinate map, agreeing with the fitted model.
        reference_ccy: Common reporting currency; FX columns exclude this zero structural factor.
        metadata_available_date: Caller-declared availability of currency/hedge/exposure metadata.
        data_available_date: Caller-declared availability of the supplied training histories.

    Result covariance approximates reference LOG excess as local LOG excess plus H times
    arithmetic forward P&L. The direct QIS reference return and nonlinear error must be
    validated downstream. No equity beta, residual policy or portfolio is silently refitted.
    """
    if not isinstance(local_model, FactorModelSnapshot) or not isinstance(spec, EquityFxRiskSpec):
        raise TypeError('Use a completed local model and EquityFxRiskSpec')
    cutoff = local_model.cutoff
    dates = [pd.Timestamp(metadata_available_date), pd.Timestamp(data_available_date)]
    if any(pd.isna(d) or d > cutoff for d in dates):
        raise ValueError('Metadata and data must be available at cutoff')
    if not isinstance(reference_ccy, str) or not reference_ccy.strip():
        raise ValueError('reference_ccy must be a nonblank currency ID')
    b = local_model.loadings
    if dict(factor_return_types) != local_model.factor_return_types:
        raise ValueError('Native factor coordinates disagree with the local fitted model')
    if local_model.provenance.get('factor_returns_freq') not in (None, spec.frequency):
        raise ValueError('Local and augmentation covariance cadences must agree')
    observed_cadences = local_model.provenance.get('asset_observation_dates', {})
    if observed_cadences and set(observed_cadences) != {spec.frequency}:
        raise ValueError('Local equity responses must use the augmentation cadence')
    panels = []
    for value, role, labels in ((factor_returns, 'factor returns', b.columns),
                                (local_log_excess_returns, 'local returns', b.index),
                                (fx_forward_payoffs, 'FX payoffs', structural_exposures.columns)):
        _dated(value, role)
        if not value.columns.equals(labels):
            raise ValueError(f'{role} labels/order must match the model and exposures')
        part = value.loc[:cutoff]
        if part.empty or part.index[-1] > dates[1]:
            raise ValueError(f'{role} observations exceed declared availability')
        if np.isinf(part.to_numpy(dtype=float)).any():
            raise ValueError(f'{role} contains infinite observations')
        panels.append(part)
    if any(not p.index.equals(panels[0].index) for p in panels[1:]):
        raise ValueError('All training panels must share the same grid; align explicitly first')
    grid = panels[0].index
    if not grid.equals(pd.date_range(grid[0], grid[-1], freq=spec.frequency)):
        raise ValueError('Training grid disagrees with frequency; retain missing rows as NaN')
    h = structural_exposures
    if (not h.index.equals(b.index) or not h.columns.is_unique or h.empty
            or reference_ccy in h.columns or not np.isfinite(h.to_numpy(dtype=float)).all()
            or any(not isinstance(c, str) or not c.strip() for c in h.columns)):
        raise ValueError('Structural exposures require exact asset order and finite non-reference FX columns')
    if set(b.columns) & {f'FX::{ccy}' for ccy in h.columns}:
        raise ValueError('Local factor names collide with the FX factor namespace')
    x, y, phi = [p.tail(spec.window) for p in panels]
    residual = y - x @ b.T
    sample = pd.concat({'factor': x, 'residual': residual, 'fx': phi}, axis=1, sort=False)
    complete = sample.dropna(how='any')
    if len(complete) < spec.min_observations:
        raise ValueError('Insufficient common observations; no filling or pairwise fallback')
    raw = _moments(complete, spec)
    _psd(raw.to_numpy(), 'estimated common-sample covariance')
    k, n = b.shape[1], b.shape[0]
    z = k + n
    joint = raw.to_numpy(copy=True)
    scale = 1.0
    if spec.covariance_policy == 'preserve_local':
        expected = b @ local_model.factor_covariance @ b.T + local_model.residual_covariance
        if not np.allclose(expected, local_model.asset_covariance, rtol=1e-10, atol=1e-12):
            raise ValueError('Supplied local Sigma must equal B F B.T plus the selected full R')
        joint[:z, :z] = 0.0
        joint[:k, :k] = local_model.factor_covariance.to_numpy()
        joint[k:z, k:z] = local_model.residual_covariance.to_numpy()
        _psd(joint[:z, :z], 'preserved local factor/residual covariance')
        cross, scale = _compatible_cross(joint[:z, :z], joint[:z, z:], joint[z:, z:],
                                         spec.cross_covariance_policy)
        joint[:z, z:] = cross
        joint[z:, :z] = cross.T
    _psd(joint, 'augmented joint covariance')
    # This is model assembly, not portfolio variance/TE analytics (owned by QIS RiskModel).
    local_projection = np.column_stack([b.to_numpy(), np.eye(n)])
    projection = np.column_stack([local_projection, h.to_numpy()])
    used_local = local_projection @ joint[:z, :z] @ local_projection.T
    covariance = projection @ joint @ projection.T
    residual_fx = joint[k:z, z:]
    omitted_term = residual_fx @ h.to_numpy().T + h.to_numpy() @ residual_fx.T
    ablated = covariance - omitted_term
    labels = b.index
    frame = lambda a: pd.DataFrame(a, index=labels, columns=labels)
    loadings = pd.concat([b, h.rename(columns=lambda c: f'FX::{c}')], axis=1, sort=False)
    types = {**dict(factor_return_types), **{f'FX::{c}': qis.ReturnTypes.DIFFERENCE for c in h.columns}}
    diagnostics = dict(
        cross_covariance_scale=scale, input_rows=len(sample), complete_rows=len(complete),
        dropped_dates=sample.index.difference(complete.index).tolist(),
        local_block_max_abs_change=float(np.max(np.abs(used_local-local_model.asset_covariance.to_numpy()))),
        residual_block_max_abs_change=float(np.max(np.abs(joint[k:z,k:z]-local_model.residual_covariance.to_numpy()))),
        raw_factor_residual_cross_max_abs=float(np.max(np.abs(raw.to_numpy()[:k,k:z]))),
        residual_fx_covariance_effect_frobenius=float(np.linalg.norm(omitted_term)),
        minimum_joint_eigenvalue=float(np.linalg.eigvalsh(joint).min()),
        minimum_ablated_asset_eigenvalue=float(np.linalg.eigvalsh(ablated).min()),
    )
    provenance = dict(
        local_model=local_model.provenance, local_residual_policy=local_model.residual_type,
        local_residual_var_weight=local_model.residual_var_weight,
        local_residual_corr_weight=local_model.residual_corr_weight,
        used_residual_policy=(local_model.residual_type if spec.covariance_policy == 'preserve_local'
                              else 'joint_empirical_full'),
        local_factor_residual_policy=('zero cross covariance imposed by supplied local model'
                                     if spec.covariance_policy == 'preserve_local'
                                     else 'estimated on common sample'),
        coordinate='reference_log_excess_first_order', local_coordinate='local_log_excess',
        fx_coordinate='arithmetic_forward_payoff_per_opening_reference_notional',
        factor_return_types=dict(factor_return_types), reference_ccy=reference_ccy,
        metadata_available_date=dates[0], data_available_date=dates[1],
        observation_date=complete.index[-1], sample_start=complete.index[0],
        frequency=spec.frequency, annualization_factor=qis.get_annualization_factor(spec.frequency),
        estimator=spec.estimator, span=spec.span if spec.estimator=='ewma' else None,
        mean_adjustment=spec.mean_adjustment, covariance_policy=spec.covariance_policy,
        cross_covariance_policy=spec.cross_covariance_policy, window=spec.window,
        missing_policy='common_complete_cases; no fills',
        structural_policy='H fixed at cutoff for next interval; embedded hedges remain in input prices',
    )
    return EquityFxRiskSnapshot(
        cutoff=cutoff, reference_ccy=reference_ccy, covariance=frame(covariance),
        without_residual_fx=frame(ablated), local_covariance=frame(used_local),
        joint_covariance=pd.DataFrame(joint,index=raw.index,columns=raw.columns),
        raw_joint_covariance=raw, loadings=loadings,
        residual_fx_covariance=pd.DataFrame(residual_fx,index=labels,columns=h.columns),
        factor_fx_covariance=pd.DataFrame(joint[:k,z:],index=b.columns,columns=h.columns),
        residual_covariance=frame(joint[k:z,k:z]), sample_index=complete.index,
        factor_return_types=types, provenance=provenance, diagnostics=diagnostics)