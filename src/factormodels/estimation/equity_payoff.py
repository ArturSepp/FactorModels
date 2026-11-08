"""Exact SIMPLE equity/FX payoff covariance for research use.

Purpose: Assemble funded equity covariance without dropping payoff interactions.
Inputs: A translated-payoff fit, arithmetic baskets/forwards and opening exposures.
Outputs: Conditional and dated historical covariance, full joint moments and QIS risk.
Units: SIMPLE P&L per opening reference notional; covariance annualised once.
Conditioning: Common complete sample; fixed forecast H and dated historical H are separate.
Ownership: FM assembles fitted risk; QIS supplies moment kernels and portfolio analytics.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd
import qis

from factormodels.analytics.models import FactorModelSnapshot
from factormodels.estimation.equity_fx import (
    EquityFxRiskSpec, _dated, _metadata, _moments, _psd, build_currency_exposure,
)

__all__ = ["EquityPayoffRiskSpec", "EquityPayoffRiskSnapshot",
           "build_funded_currency_exposure", "compute_payoff_factor_returns",
           "assemble_equity_payoff_risk"]


@dataclass(frozen=True)
class EquityPayoffRiskSpec:
    """Common moments for SIMPLE payoffs; every estimated cross block is retained.

    Args:
        frequency: Complete observation grid and covariance annualisation cadence.
        window: Trailing grid rows before a single complete-case mask.
        min_observations: Minimum complete observations, with no filling.
        estimator: QIS sample or EWMA covariance kernel.
        span: EWMA span, in observations.
        mean_adjustment: Sample centering, causal EWMA centering or no EWMA centering.
    """
    frequency: str
    window: int = 60
    min_observations: int = 36
    estimator: Literal["sample", "ewma"] = "sample"
    span: int = 36
    mean_adjustment: Literal["sample", "ewma", "none"] = "sample"

    def __post_init__(self) -> None:
        """Apply the shared cadence/kernel validation without a marginal-preservation mode."""
        EquityFxRiskSpec(self.frequency, "joint_reestimate", self.window,
                        self.min_observations, self.estimator, self.span, self.mean_adjustment)


def build_funded_currency_exposure(
    local_ccys: pd.Series | pd.DataFrame,
    hedge_ratios: float | pd.Series | pd.DataFrame,
    local_cash_returns: pd.Series,
    currencies: list[str] | pd.Index, *, reference_ccy: str, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return H[i,c] = indicator(i,c) * (1 + local cash[i] - hedge[i]).

    Args:
        local_ccys: Quote-currency IDs, static or available as of cutoff.
        hedge_ratios: Additional opening-principal hedges, selected as of cutoff.
        local_cash_returns: Asset-labelled SIMPLE cash for the declared next interval,
            accrued from quotes known at cutoff; these are not annual rate quotes.
        currencies: Ordered foreign currency factors, excluding the reference ID.
        reference_ccy: Reporting currency with zero direct structural FX exposure.
        cutoff: Opening information boundary; no future currency/hedge row is used.

    Cash timing and quote availability belong to the caller, normally the QIS FX API.
    The funded return identity assumes synthetic CIP forwards with matching cash legs.
    """
    base = build_currency_exposure(local_ccys, hedge_ratios, currencies,
                                   reference_ccy=reference_ccy, cutoff=cutoff)
    if not isinstance(local_cash_returns, pd.Series):
        raise TypeError("local_cash_returns must be an asset-labelled Series")
    cash = _metadata(local_cash_returns, pd.Timestamp(cutoff), base.index, "local cash")
    cash = cash.astype(float)
    if not np.isfinite(cash).all() or (cash <= -1.0).any():
        raise ValueError("Local cash must be finite with positive gross growth")
    incidence = build_currency_exposure(local_ccys, 0.0, currencies,
                                        reference_ccy=reference_ccy, cutoff=cutoff)
    return base + incidence.mul(cash, axis=0)


def compute_payoff_factor_returns(
    translated_payoffs: pd.DataFrame, opening_weights: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Form arithmetic baskets of translated SIMPLE payoffs, without synthetic NAVs.

    Args:
        translated_payoffs: Asset payoffs per opening reference notional on a complete grid.
        opening_weights: One identically labelled panel per basket. Each row is the weight
            known at that interval's opening, not a closing-date decision. The caller declares
            market weight sum, style gross notional, neutrality and reset policies.

    Every constituent must be observed; missing payoffs are not silently zeroed or renormalised.
    """
    _dated(translated_payoffs, "translated payoffs")
    if not opening_weights:
        raise ValueError("At least one payoff basket is required")
    result = {}
    for name, weights in opening_weights.items():
        _dated(weights, "opening basket weights")
        if (not isinstance(name, str) or not name.strip()
                or not weights.index.equals(translated_payoffs.index)
                or not weights.columns.equals(translated_payoffs.columns)
                or not np.isfinite(weights.to_numpy()).all()):
            raise ValueError("Basket weights require finite values and exact date/asset order")
        result[name] = (translated_payoffs * weights).sum(axis=1, min_count=len(weights.columns))
    return pd.DataFrame(result, index=translated_payoffs.index)


@dataclass(frozen=True, init=False)
class EquityPayoffRiskSnapshot:
    """Owned research moments; forecast covariance and historical reconstruction are distinct.

    No diagonal residual decomposition is exported: the authoritative covariance includes
    factor/residual, factor/FX and residual/FX dependence. Diagnostic ablations are not models.
    """
    cutoff: pd.Timestamp
    reference_ccy: str
    _fields: dict

    def __init__(self, *, cutoff: pd.Timestamp, reference_ccy: str, fields: dict) -> None:
        """Own the completed fields and validate both covariance projections."""
        for key in ("covariance", "historical_covariance", "joint_covariance"):
            _psd(fields[key].to_numpy(), key)
        object.__setattr__(self, "cutoff", pd.Timestamp(cutoff))
        object.__setattr__(self, "reference_ccy", reference_ccy)
        object.__setattr__(self, "_fields", deepcopy(fields))

    @property
    def covariance(self) -> pd.DataFrame:
        """Return annual forecast covariance with H fixed at the cutoff."""
        return deepcopy(self._fields["covariance"])

    @property
    def historical_covariance(self) -> pd.DataFrame:
        """Return descriptive covariance using each historical interval's actual opening H."""
        return deepcopy(self._fields["historical_covariance"])

    @property
    def joint_covariance(self) -> pd.DataFrame:
        """Return annual joint moments of factors, residuals, FX and dated FX contributions."""
        return deepcopy(self._fields["joint_covariance"])

    @property
    def loadings(self) -> pd.DataFrame:
        """Return the fitted payoff B and fixed forecast H, with labelled FX columns."""
        return deepcopy(self._fields["loadings"])

    @property
    def reconstructed_returns(self) -> pd.DataFrame:
        """Return historical SIMPLE reference excess from p plus dated H times phi."""
        return deepcopy(self._fields["reconstructed_returns"])

    @property
    def sample_index(self) -> pd.DatetimeIndex:
        """Return the one complete observation mask used by every moment."""
        return self._fields["sample_index"].copy()

    @property
    def without_residual_fx(self) -> pd.DataFrame:
        """Return the fixed-H omission diagnostic; it can be indefinite."""
        return deepcopy(self._fields["without_residual_fx"])

    @property
    def provenance(self) -> dict:
        """Return units, timing, original fit policy and common-sample choices."""
        return deepcopy(self._fields["provenance"])

    @property
    def diagnostics(self) -> dict:
        """Return PSD and cross-block diagnostics without silently altering moments."""
        return deepcopy(self._fields["diagnostics"])

    def to_risk_model(self, *, include_residual_fx: bool = True) -> qis.RiskModel:
        """Export fixed-H covariance to QIS, rejecting an indefinite omission diagnostic."""
        covariance = self.covariance if include_residual_fx else self.without_residual_fx
        _psd(covariance.to_numpy(), "selected payoff covariance")
        return qis.RiskModel(covar={self.cutoff: covariance},
                             factor_loadings={self.cutoff: self.loadings},
                             factor_return_types={c: qis.ReturnTypes.DIFFERENCE
                                                  for c in self.loadings.columns})


# QIS RiskModel consumes covariance; it does not assemble fitted payoff/currency cross blocks.
def assemble_equity_payoff_risk(
    payoff_model: FactorModelSnapshot, factor_payoffs: pd.DataFrame,
    translated_payoffs: pd.DataFrame, fx_forward_payoffs: pd.DataFrame,
    forecast_exposures: pd.DataFrame, historical_exposures: Mapping[pd.Timestamp, pd.DataFrame],
    *, spec: EquityPayoffRiskSpec, reference_ccy: str, forecast_end: pd.Timestamp,
    metadata_available_date: pd.Timestamp, data_available_date: pd.Timestamp,
) -> EquityPayoffRiskSnapshot:
    """Assemble exact SIMPLE payoff risk with full joint cross covariance.

    Args:
        payoff_model: Completed fit of translated local arithmetic excess p on payoff baskets.
            All basket types must be DIFFERENCE (increments of cumulative arithmetic P&L).
        factor_payoffs: Arithmetic basket histories x, in fitted factor order.
        translated_payoffs: QIS LOCAL_EXCESS_TRANSLATED / SIMPLE responses p.
        fx_forward_payoffs: Arithmetic long-local synthetic CIP forward payoffs phi.
        forecast_exposures: H fixed at cutoff for the interval ending at forecast_end.
        historical_exposures: Opening-known H for each end-labelled historical observation.
        spec: One sample, centering, estimator and annualisation for all covariance blocks.
        reference_ccy: Reporting currency; omitted from the FX factor columns.
        forecast_end: Next valuation date for the cash horizon used to construct forecast H.
        metadata_available_date: Caller declaration, no later than the fit cutoff.
        data_available_date: Caller declaration, covering observations through the cutoff.

    B is fixed from the completed fit; residuals are recomputed as p-Bx on the selected sample.
    Full common-sample moments replace the fit's selected marginals explicitly. There is no
    orthogonality assumption, cross-block clipping or log-payoff conversion. The forecast map
    is exact for specified moments, not a claim of forecast accuracy. A separate projection
    using the dated FX contributions validates historical covariance without averaging H.
    """
    if not isinstance(payoff_model, FactorModelSnapshot) or not isinstance(spec, EquityPayoffRiskSpec):
        raise TypeError("Use a completed payoff model and EquityPayoffRiskSpec")
    cutoff = payoff_model.cutoff
    availability = [pd.Timestamp(metadata_available_date), pd.Timestamp(data_available_date)]
    end = pd.Timestamp(forecast_end)
    if any(pd.isna(d) or d > cutoff for d in availability) or pd.isna(end) or end <= cutoff:
        raise ValueError("Inputs must be available at cutoff and forecast_end must follow it")
    if not isinstance(reference_ccy, str) or not reference_ccy.strip():
        raise ValueError("reference_ccy must be a nonblank ID")
    b = payoff_model.loadings
    if set(payoff_model.factor_return_types) != set(b.columns) or any(
            t != qis.ReturnTypes.DIFFERENCE for t in payoff_model.factor_return_types.values()):
        raise ValueError("Payoff baskets require DIFFERENCE coordinates, never LOG")
    provenance = payoff_model.provenance
    if provenance.get("asset_return_type") != "simple_payoff":
        raise ValueError("The completed fit must declare asset_return_type=simple_payoff")
    if (provenance.get("factor_returns_freq") not in (None, spec.frequency)
            or set(provenance.get("asset_observation_dates", {spec.frequency: cutoff}))
            != {spec.frequency}):
        raise ValueError("Payoff fit and covariance must share one cadence")
    ids = fx_forward_payoffs.columns
    if (not ids.is_unique or ids.empty or reference_ccy in ids
            or any(not isinstance(c, str) or not c.strip() for c in ids)
            or set(b.columns) & {f"FX::{c}" for c in ids}):
        raise ValueError("FX factors require unique foreign IDs without factor-name collisions")

    def exposure(value: pd.DataFrame) -> np.ndarray:
        """Check exact exposure order; do not silently reindex missing assets or currencies."""
        if (not isinstance(value, pd.DataFrame) or not value.index.equals(b.index)
                or not value.columns.equals(ids) or not np.isfinite(value.to_numpy()).all()):
            raise ValueError("Exposure matrices require exact labels/order and finite values")
        return value.to_numpy(dtype=float)

    h = exposure(forecast_exposures)
    panels = []
    for panel, labels in ((factor_payoffs, b.columns), (translated_payoffs, b.index),
                          (fx_forward_payoffs, ids)):
        _dated(panel, "payoff history")
        part = panel.loc[:cutoff]
        if (not panel.columns.equals(labels) or part.empty
                or part.index[-1] > availability[1] or np.isinf(part.to_numpy()).any()):
            raise ValueError("Histories require matching labels and finite available observations")
        panels.append(part.tail(spec.window))
    grid = panels[0].index
    if (any(not p.index.equals(grid) for p in panels[1:])
            or not grid.equals(pd.date_range(grid[0], grid[-1], freq=spec.frequency))):
        raise ValueError("All histories must retain the same complete declared grid")
    x, p, phi = panels
    historical = []
    for date in grid:
        if date not in historical_exposures:
            raise ValueError(f"Missing dated opening exposure for {date}")
        historical.append(phi.loc[date].to_numpy() @ exposure(historical_exposures[date]).T)
    g = pd.DataFrame(historical, index=grid, columns=b.index)
    residual = p - x @ b.T
    sample = pd.concat({"factor": x, "residual": residual, "fx": phi,
                        "historical_fx": g}, axis=1).dropna(how="any")
    if len(sample) < spec.min_observations:
        raise ValueError("Insufficient common observations; no filling or pairwise fallback")
    joint = _moments(sample, spec)
    _psd(joint.to_numpy(), "full payoff joint covariance")
    k, n, m = b.shape[1], b.shape[0], len(ids)
    z = k + n
    matrix = joint.to_numpy()
    # Model projection, not portfolio variance/TE analytics (owned by QIS RiskModel).
    base = np.column_stack([b.to_numpy(), np.eye(n)])
    fixed = np.column_stack([base, h, np.zeros((n, n))])
    dated = np.column_stack([base, np.zeros((n, m)), np.eye(n)])
    covariance = fixed @ matrix @ fixed.T
    historical_covariance = dated @ matrix @ dated.T
    c = matrix[k:z, z:z+m]
    omission = c @ h.T + h @ c.T
    cx = matrix[:k, k:z]
    factor_residual = b.to_numpy() @ cx + cx.T @ b.to_numpy().T
    loadings = pd.concat([b, forecast_exposures.rename(columns=lambda c: f"FX::{c}")], axis=1)

    def frame(values: np.ndarray) -> pd.DataFrame:
        """Label an asset covariance without changing its order."""
        return pd.DataFrame(values, index=b.index, columns=b.index)

    fields = dict(
        covariance=frame(covariance), historical_covariance=frame(historical_covariance),
        without_residual_fx=frame(covariance-omission), joint_covariance=joint,
        loadings=loadings, reconstructed_returns=(p+g).loc[sample.index], sample_index=sample.index,
        diagnostics=dict(input_rows=len(grid), complete_rows=len(sample),
            dropped_dates=grid.difference(sample.index).tolist(),
            minimum_joint_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
            residual_fx_effect_frobenius=float(np.linalg.norm(omission)),
            factor_residual_effect_frobenius=float(np.linalg.norm(factor_residual)),
            raw_factor_residual_cross_max_abs=float(np.abs(cx).max()),
            fixed_vs_dated_covariance_max_abs=float(np.abs(covariance-historical_covariance).max())),
        provenance=dict(reference_ccy=reference_ccy, research_only=True,
            coordinate="reference_simple_excess", response_coordinate="translated_local_excess_simple",
            factor_coordinate="arithmetic_payoff_per_opening_reference_notional",
            forecast_end=end, metadata_available_date=availability[0], data_available_date=availability[1],
            sample_start=sample.index[0], observation_date=sample.index[-1], frequency=spec.frequency,
            annualization_factor=qis.get_annualization_factor(spec.frequency),
            estimator=spec.estimator, span=spec.span if spec.estimator == "ewma" else None,
            mean_adjustment=spec.mean_adjustment, window=spec.window,
            covariance_policy="joint_reestimate_all_blocks", residual_policy="full_empirical",
            original_fit=provenance, original_residual_policy=payoff_model.residual_type,
            missing_policy="one_common_complete_case_mask", forward_policy="synthetic_CIP",
            forecast_policy="fixed opening H for next interval", historical_policy="dated opening H"))
    return EquityPayoffRiskSnapshot(cutoff=cutoff, reference_ccy=reference_ccy, fields=fields)
