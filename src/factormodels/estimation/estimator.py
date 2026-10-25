"""Current and rolling factor-model estimation from supplied market histories.
Purpose: Fit complete current and rolling factor models from supplied historical data.
Inputs: FL configuration, factor prices, per-cadence asset returns, cutoff and residual policy.
Outputs: FactorCovarEstimator.fit_current and fit_rolling return completed FM contracts.
Units: Asset observations are log returns; factor units are declared and output moments annualised by native cadence.
Conditioning: Truncate all histories at each cutoff, isolate fitting state and preserve the selected residual covariance.
Ownership: FM orchestrates and owns completed results; FL fits loadings/residuals and QIS estimates factor covariance.
"""
from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from datetime import date, datetime
from factorlasso import CurrentFactorCovarData

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel

from factormodels.estimation._pipeline import _FactorCovarPipeline
from factormodels.estimation._results import _materialize
from factormodels.analytics.models import (
    DatedFactorCovariance, FactorModelSnapshot, RollingFactorModel, _return_types,
)


__all__ = ['FactorCovarEstimator']


def _historical_panel(panel: pd.DataFrame, cutoff: pd.Timestamp, name: str) -> pd.DataFrame:
    """Validate labels and return an owned history ending at the requested cutoff."""
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if (not isinstance(panel.index, pd.DatetimeIndex) or panel.index.hasnans
            or not panel.index.is_unique or not panel.index.is_monotonic_increasing):
        raise ValueError(f"{name} requires an ordered unique DatetimeIndex")
    if not panel.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    result = panel.loc[:cutoff].copy()
    if result.empty:
        raise ValueError(f"{name} has no observations at cutoff")
    return result


@dataclass(frozen=True)
class FactorCovarEstimator:
    """Configure a complete current/rolling factor fit without mutating supplied models.

    Factor covariance uses QIS sampling/EWMA; loadings, priors, signs, clustering
    and residual statistics use FactorLasso. Every fit returns a completed FM
    model, including its selected asset covariance. Configuration is separate
    from fitted state; use dataclasses.replace to construct another configuration.

    Args:
        lasso_model: FL configuration, copied without fitted state for each call.
        factor_returns_freq: Factor-covariance sampling cadence, separate from the
            regression cadences in asset_returns_dict.
        factor_covar_span: EWMA span in factor return observations.
        rebalancing_freq: Calendar cadence for rolling output snapshots.
        is_apply_vol_normalised_returns: Select the QIS normalized-return kernel.
        include_factors_in_clustering: Use factor references for HCGL/FCGL partitions.
        factor_clustering_freqs: Selected response cadences for those references.
        residual_type: orthogonal, empirical, exposure_cluster or residual_cluster.
        residual_covar_freq: Common residual grid; None selects the slowest cadence.
        residual_covar_span: EWMA span on the common residual grid.
        residual_corr_weight: Correlation retention for nonorthogonal residual risk.
        residual_var_weight: Multiplier on the complete selected residual block.
        reg_lambda_freq_dict: Optional fixed regression penalty per response cadence.
        factor_return_types: Exact factor coordinate map; None means LOG.
        factor_groups: Optional provider-neutral QIS factor groups.

    Internal factor covariance is demeaned using the existing QIS EWMA contract.
    Regression demeaning remains a separate LassoModel choice. Asset return
    observations must be supplied in log-return units on their declared cadence.
    """

    lasso_model: LassoModel
    factor_returns_freq: str = "W-WED"
    factor_covar_span: int = 52
    rebalancing_freq: str = "QE"
    is_apply_vol_normalised_returns: bool = False
    include_factors_in_clustering: bool = False
    factor_clustering_freqs: list[str] | None = None
    residual_type: str = "orthogonal"
    residual_covar_freq: str | None = None
    residual_covar_span: float | None = None
    residual_corr_weight: float = 1.
    residual_var_weight: float = 1.
    reg_lambda_freq_dict: dict[str, float] | None = None
    factor_return_types: dict | None = None
    factor_groups: dict | None = None

    def __post_init__(self) -> None:
        """Validate configuration before a numerical fit is attempted."""
        if not isinstance(self.lasso_model, LassoModel):
            raise TypeError("lasso_model must be a FactorLasso LassoModel configuration")
        if not np.isfinite(self.residual_var_weight) or self.residual_var_weight < 0:
            raise ValueError("residual_var_weight must be finite and nonnegative")
        self._pipeline()

    def _pipeline(self) -> _FactorCovarPipeline:
        """Create isolated fitting state from configuration only."""
        return _FactorCovarPipeline(
            lasso_model=LassoModel(**deepcopy(self.lasso_model.get_params())),
            factor_returns_freq=self.factor_returns_freq,
            factor_covar_span=self.factor_covar_span,
            rebalancing_freq=self.rebalancing_freq,
            is_apply_vol_normalised_returns=self.is_apply_vol_normalised_returns,
            include_factors_in_clustering=self.include_factors_in_clustering,
            factor_clustering_freqs=deepcopy(self.factor_clustering_freqs),
            residual_type=self.residual_type,
            residual_covar_freq=self.residual_covar_freq,
            residual_covar_span=self.residual_covar_span,
            residual_corr_weight=self.residual_corr_weight,
            reg_lambda_freq_dict=deepcopy(self.reg_lambda_freq_dict),
            factor_return_types=deepcopy(self.factor_return_types),
        )

    def _inputs(self, risk_factor_prices: pd.DataFrame, asset_returns_dict: Mapping[str, pd.DataFrame], cutoff: str | date | datetime | np.datetime64 | pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame, dict[str, pd.DataFrame]]:
        """Truncate every historical panel and validate the asset partition."""
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff):
            raise ValueError("cutoff must be a valid date")
        factors = _historical_panel(risk_factor_prices, cutoff, "risk_factor_prices")
        if not asset_returns_dict:
            raise ValueError("asset_returns_dict must contain at least one cadence")
        returns = {freq: _historical_panel(panel, cutoff, f"returns[{freq}]")
                   for freq, panel in asset_returns_dict.items()}
        labels = [asset for panel in returns.values() for asset in panel.columns]
        if len(set(labels)) != len(labels):
            raise ValueError("assets must belong to exactly one return cadence")
        return cutoff, factors, returns

    def _snapshot(self, data: CurrentFactorCovarData, cutoff: pd.Timestamp, provenance: Mapping[str, Any]) -> FactorModelSnapshot:
        """Attach selected assembly policy and coordinate metadata to a fitted result."""
        estimated = provenance["factor_covariance"] == "estimated"
        provenance = {
            **provenance,
            "factor_returns_freq": self.factor_returns_freq if estimated else None,
            "factor_covar_span": self.factor_covar_span if estimated else None,
            "factor_covariance_units": "annual_native_coordinates",
            "factor_covariance_demean": True if estimated else None,
            "factor_covariance_vol_normalised": (
                self.is_apply_vol_normalised_returns if estimated else None),
            "asset_return_type": "log",
            "reg_lambda": self.lasso_model.reg_lambda,
            "reg_lambda_freq_dict": deepcopy(self.reg_lambda_freq_dict),
        }
        return _materialize(
            data=data, cutoff=cutoff, residual_type=self.residual_type,
            residual_var_weight=self.residual_var_weight,
            residual_corr_weight=self.residual_corr_weight,
            factor_return_types=self.factor_return_types,
            factor_groups=self.factor_groups, provenance=provenance,
        )

    def fit_current(
        self, risk_factor_prices: pd.DataFrame,
        asset_returns_dict: dict[str, pd.DataFrame], *,
        cutoff: pd.Timestamp, assets: list[str] | pd.Index | None = None,
        factor_covariance: DatedFactorCovariance | None = None,
        precomputed_clusters: dict | None = None,
        precomputed_linkages: dict | None = None,
        precomputed_cutoffs: dict | None = None,
        precomputed_asof: pd.Timestamp | None = None,
    ) -> FactorModelSnapshot:
        """Fit one completed model using only information available through cutoff.

        A supplied covariance must declare observation and availability dates.
        External clustering maps require precomputed_asof and all three maps;
        this date declares when the assignments were available. Priors/signs in
        lasso_model are caller-selected information that must also be known by
        cutoff. Input indices declare availability; FM cannot infer publication
        delays or restatements absent from the supplied data.
        """
        cutoff, factors, returns = self._inputs(risk_factor_prices, asset_returns_dict, cutoff)
        return_types = _return_types(factors.columns, self.factor_return_types)
        covariance = None
        provenance = {
            "factor_observation_date": factors.index[-1],
            "asset_observation_dates": {freq: panel.index[-1] for freq, panel in returns.items()},
            "factor_covariance": "estimated",
        }
        if factor_covariance is not None:
            if not isinstance(factor_covariance, DatedFactorCovariance):
                raise TypeError("factor_covariance must be DatedFactorCovariance")
            covariance = factor_covariance.at_cutoff(cutoff, factors.columns, return_types)
            provenance.update(
                factor_covariance="supplied",
                covariance_observation_date=factor_covariance.observation_date,
                covariance_available_date=factor_covariance.available_date,
            )
        if any(value is not None for value in (
                precomputed_clusters, precomputed_linkages, precomputed_cutoffs)):
            if precomputed_asof is None or pd.isna(pd.Timestamp(precomputed_asof)):
                raise ValueError("external clustering requires precomputed_asof")
            if pd.Timestamp(precomputed_asof) > cutoff:
                raise ValueError("external clustering is not available at cutoff")
            provenance["clustering_available_date"] = pd.Timestamp(precomputed_asof)
        pipeline = self._pipeline()
        data = pipeline.fit_current_factor_covars(
            risk_factor_prices=factors, asset_returns_dict=returns,
            assets=assets, x_covar=covariance, estimation_date=cutoff,
            precomputed_clusters=deepcopy(precomputed_clusters),
            precomputed_linkages=deepcopy(precomputed_linkages),
            precomputed_cutoffs=deepcopy(precomputed_cutoffs),
        )
        return self._snapshot(data, cutoff, provenance)

    def fit_rolling(
        self, risk_factor_prices: pd.DataFrame,
        asset_returns_dict: dict[str, pd.DataFrame], *,
        time_period: qis.TimePeriod, assets: list[str] | pd.Index | None = None,
        rebalancing_freq: str | None = None,
    ) -> RollingFactorModel:
        """Fit completed expanding-history models on the selected calendar schedule.

        Each snapshot is causal at its own date. Cluster smoothing is prepared by
        FactorLasso on the same causal schedule; complete common-period residual
        correlation may be held between slower residual observations.
        """
        _, factors, returns = self._inputs(risk_factor_prices, asset_returns_dict, time_period.end)
        _return_types(factors.columns, self.factor_return_types)
        pipeline = self._pipeline()
        data = pipeline.fit_rolling_factor_covars(
            risk_factor_prices=factors, asset_returns_dict=returns,
            time_period=time_period, assets=assets, rebalancing_freq=rebalancing_freq,
        )
        return RollingFactorModel({
            date: self._snapshot(fitted, date, {
                "factor_observation_date": factors.loc[:date].index[-1],
                "asset_observation_dates": {
                    freq: panel.loc[:date].index[-1] for freq, panel in returns.items()},
                "factor_covariance": "estimated",
            })
            for date, fitted in data.data.items()
        })
