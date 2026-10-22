"""Versioned six-file factor inputs: load, fit and persist are separate actions.

No function fetches data. ``load_bundle`` only reads supplied files;
``fit_bundle`` works in memory; ``save_bundle`` creates a new directory.
The schema stores data coordinates and configuration, not executable recipes.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType
from qis.perfstats.config import ReturnTypes

from factormodels.utils.local_path import create_output_directory, resolve_resource_path
from factormodels.estimation import FactorCovarEstimator
from factormodels.model import FactorModelSnapshot, RollingFactorModel

__all__ = ["BUNDLE_SCHEMA_VERSION", "BundleSettings", "FactorBundle", "load_bundle", "fit_bundle", "save_bundle"]

BUNDLE_SCHEMA_VERSION = "1"
_FILES = {
    "factor_levels": "futures_risk_factors.csv",
    "fx_spots": "fx_hedging_data_fx_spots.csv",
    "domestic_rates": "fx_hedging_data_domestic_rates.csv",
    "asset_prices": "asset_prices.csv",
    "asset_metadata": "asset_metadata.csv",
    "settings": "risk_model_settings.csv",
}
_BOOLS = {"is_log_returns", "is_excess_returns", "demean", "is_apply_vol_normalised_returns",
          "include_factors_in_clustering", "zero_return_to_nan"}
_INTS = {"factor_covar_span", "beta_span", "warmup_period", "cash_rate_lag"}
_FLOATS = {"reg_lambda", "residual_corr_weight", "residual_var_weight"}
_JSON = {"factor_return_types", "factor_units", "span_freq_dict", "reg_lambda_freq_dict",
         "factor_clustering_freqs", "residual_covar_span", "residual_covar_freq"}


def _frequency(value: str, name: str) -> None:
    """Require a positive pandas cadence, without silently changing its spelling."""
    try:
        offset = pd.tseries.frequencies.to_offset(value)
        if offset.n <= 0:
            raise ValueError("nonpositive cadence")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid positive frequency") from exc


@dataclass(frozen=True, kw_only=True)
class BundleSettings:
    """Complete, inspectable calibration and units for one supplied bundle.

    Args:
        reference_ccy: Currency of factor NAVs and converted asset returns.
        factor_names: Exact ordered factor columns.
        estimation_start: First rolling output date; earlier history is warm-up data.
        estimation_end: Information cutoff for current and rolling fits.
        recipe_id: Caller-declared identity of the supplied factor construction.
        factor_return_types: Exact factor map of ``LOG`` or ``DIFFERENCE``;
            omitted means LOG for every factor.
        factor_units: Native level/return units; LOG factors use ``dimensionless``.
            DIFFERENCE factors require an explicit native unit, such as ``index_points``.
        factor_return_basis: ``total`` or ``excess``, agreeing with asset return basis.
        schema_version: The single supported version, ``1``.

    Other fields correspond to named FactorLasso/FactorCovarEstimator settings.
    Rates use annual decimal fractions and FX quotes use USD per currency unit.
    Covariances are annual in the factors' declared native coordinates. Settings
    are frozen; mappings are copied and read-only. Construct variants with
    ``dataclasses.replace``. No persisted historical format is inferred.
    """

    reference_ccy: str
    factor_names: tuple[str, ...]
    estimation_start: pd.Timestamp
    estimation_end: pd.Timestamp
    recipe_id: str
    schema_version: str = BUNDLE_SCHEMA_VERSION
    is_log_returns: bool = True
    is_excess_returns: bool = False
    zero_return_to_nan: bool = True
    cash_rate_lag: int = 1
    factor_returns_freq: str = "ME"
    factor_covar_span: int = 36
    rebalancing_freq: str = "QE"
    lasso_model_type: str = "LASSO"
    reg_lambda: float = 1.e-5
    beta_span: int = 36
    warmup_period: int = 6
    demean: bool = True
    solver: str = "CLARABEL"
    loss_normalization: str = "sample"
    span_freq_dict: Mapping[str, float] | None = None
    reg_lambda_freq_dict: Mapping[str, float] | None = None
    factor_return_types: Mapping[str, str] | None = None
    factor_units: Mapping[str, str] | None = None
    factor_return_basis: str = "total"
    rates_unit: str = "annual_decimal"
    fx_spot_convention: str = "USD_per_currency"
    covariance_units: str = "annual_native_coordinates"
    is_apply_vol_normalised_returns: bool = False
    include_factors_in_clustering: bool = False
    factor_clustering_freqs: tuple[str, ...] | None = None
    residual_type: str = "orthogonal"
    residual_covar_freq: str | None = None
    residual_covar_span: float | None = None
    residual_corr_weight: float = 1.
    residual_var_weight: float = 1.

    def __post_init__(self) -> None:
        """Validate units, chronology and the supported estimator configuration."""
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version!r}; expected '1'")
        if isinstance(self.factor_names, str):
            raise ValueError("factor_names must be an ordered sequence, not a string")
        names = tuple(self.factor_names)
        if not names or len(names) != len(set(names)) or any(
                not isinstance(name, str) or not name.strip() or name != name.strip() or "|" in name
                for name in names):
            raise ValueError("factor_names must contain nonempty unique trimmed labels without '|'")
        object.__setattr__(self, "factor_names", names)
        for name in ("reference_ccy", "recipe_id", "solver"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a nonempty trimmed string")
        for name in _BOOLS:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not self.is_log_returns:
            raise ValueError("asset is_log_returns must be true for FactorCovarEstimator")
        if self.factor_return_basis != ("excess" if self.is_excess_returns else "total"):
            raise ValueError("factor_return_basis must agree with is_excess_returns")
        for name, expected in (("rates_unit", "annual_decimal"),
                               ("fx_spot_convention", "USD_per_currency"),
                               ("covariance_units", "annual_native_coordinates")):
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")
        for name in ("estimation_start", "estimation_end"):
            value = pd.Timestamp(getattr(self, name))
            if pd.isna(value) or value.tz is not None:
                raise ValueError(f"{name} must be a valid timezone-naive date")
            object.__setattr__(self, name, value)
        if self.estimation_start > self.estimation_end:
            raise ValueError("estimation_start must not be after estimation_end")
        for name in _INTS:
            value = getattr(self, name)
            minimum = 0 if name == "cash_rate_lag" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be a {'nonnegative' if minimum == 0 else 'positive'} integer")
        for name in _FLOATS:
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.loss_normalization not in {"sample", "weight_sum"}:
            raise ValueError("loss_normalization must be sample or weight_sum")
        if self.lasso_model_type not in LassoModelType.__members__:
            raise ValueError(f"unknown lasso_model_type={self.lasso_model_type!r}")
        for name in ("factor_returns_freq", "rebalancing_freq", "residual_covar_freq"):
            if getattr(self, name) is not None:
                _frequency(getattr(self, name), name)
        if self.residual_covar_span is not None and (
                not np.isfinite(self.residual_covar_span) or self.residual_covar_span <= 0):
            raise ValueError("residual_covar_span must be finite and positive")
        if self.factor_clustering_freqs is not None:
            for value in self.factor_clustering_freqs:
                _frequency(value, "factor_clustering_freqs")
            object.__setattr__(self, "factor_clustering_freqs", tuple(self.factor_clustering_freqs))
        for name in ("span_freq_dict", "reg_lambda_freq_dict"):
            mapping = getattr(self, name)
            if mapping is not None:
                if not isinstance(mapping, Mapping):
                    raise ValueError(f"{name} must be a frequency-to-value mapping")
                for cadence, value in mapping.items():
                    _frequency(cadence, name)
                    if not np.isfinite(value) or value < 0 or (name == "span_freq_dict" and value == 0):
                        raise ValueError(f"{name} has an invalid value")
                object.__setattr__(self, name, MappingProxyType(dict(mapping)))
        if self.factor_return_types is not None and not isinstance(self.factor_return_types, Mapping):
            raise ValueError("factor_return_types must be a mapping")
        transforms = dict(dict.fromkeys(names, "LOG") if self.factor_return_types is None else self.factor_return_types)
        if set(transforms) != set(names) or any(value not in {"LOG", "DIFFERENCE"} for value in transforms.values()):
            raise ValueError("factor_return_types must map every factor exactly to LOG or DIFFERENCE")
        if self.factor_units is not None and not isinstance(self.factor_units, Mapping):
            raise ValueError("factor_units must be a mapping")
        units = dict({name: "dimensionless" for name in names if transforms[name] == "LOG"}
                     if self.factor_units is None else self.factor_units)
        if set(units) != set(names) or any(not isinstance(value, str) or not value.strip() for value in units.values()):
            raise ValueError("factor_units must declare every factor's native unit")
        for name in names:
            if (transforms[name] == "LOG") != (units[name] == "dimensionless"):
                raise ValueError("LOG requires dimensionless units; DIFFERENCE requires explicit native units")
        object.__setattr__(self, "factor_return_types", MappingProxyType(transforms))
        object.__setattr__(self, "factor_units", MappingProxyType(units))
        self.make_estimator()

    def make_estimator(self) -> FactorCovarEstimator:
        """Construct fresh FL configuration and the Stage-3 FM estimator, without fitting."""
        return FactorCovarEstimator(
            lasso_model=LassoModel(
                model_type=LassoModelType[self.lasso_model_type], reg_lambda=self.reg_lambda,
                span=self.beta_span, warmup_period=self.warmup_period, demean=self.demean,
                solver=self.solver, loss_normalization=self.loss_normalization,
                span_freq_dict=None if self.span_freq_dict is None else dict(self.span_freq_dict)),
            factor_returns_freq=self.factor_returns_freq, factor_covar_span=self.factor_covar_span,
            rebalancing_freq=self.rebalancing_freq, residual_type=self.residual_type,
            residual_covar_freq=self.residual_covar_freq, residual_covar_span=self.residual_covar_span,
            residual_corr_weight=self.residual_corr_weight, residual_var_weight=self.residual_var_weight,
            reg_lambda_freq_dict=None if self.reg_lambda_freq_dict is None else dict(self.reg_lambda_freq_dict),
            is_apply_vol_normalised_returns=self.is_apply_vol_normalised_returns,
            include_factors_in_clustering=self.include_factors_in_clustering,
            factor_clustering_freqs=None if self.factor_clustering_freqs is None else list(self.factor_clustering_freqs),
            factor_return_types={name: ReturnTypes[value] for name, value in self.factor_return_types.items()},
        )

    def to_frame(self) -> pd.DataFrame:
        """Return all schema settings as a human-editable setting/value CSV frame."""
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "factor_names":
                value = "|".join(value)
            elif field.name in _JSON:
                value = json.dumps(dict(value) if isinstance(value, Mapping) else value, allow_nan=False)
            elif isinstance(value, pd.Timestamp):
                value = value.isoformat()
            values[field.name] = str(value)
        return pd.Series(values, name="value").to_frame().rename_axis("setting")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> BundleSettings:
        """Read exactly this version's fields; reject duplicates, unknowns and omissions."""
        if list(frame.columns) != ["value"] or not frame.index.is_unique:
            raise ValueError("settings require unique setting rows and exactly one value column")
        values = frame["value"].to_dict()
        if values.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported or missing schema_version; expected '1'")
        required = {field.name for field in fields(cls)}
        if set(values) != required:
            raise ValueError(f"settings mismatch: missing={sorted(required - set(values))}, unknown={sorted(set(values) - required)}")
        for name, value in values.items():
            if name in _BOOLS:
                if str(value).lower() not in {"true", "false"}:
                    raise ValueError(f"{name} must be a true/false boolean")
                values[name] = str(value).lower() == "true"
            elif name in _INTS:
                if not str(value).isdigit():
                    raise ValueError(f"{name} must be a positive integer")
                values[name] = int(value)
            elif name in _FLOATS:
                values[name] = float(value)
            elif name in _JSON:
                values[name] = json.loads(value)
            elif name == "factor_names":
                values[name] = str(value).split("|")
        return cls(**values)


@dataclass(frozen=True)
class FactorBundle:
    """Validated supplied histories, with owned mutable DataFrame copies.

    Args:
        factor_levels: Ordered NAVs or typed observed levels, already on reference-currency basis.
        fx_spots: Positive USD per currency-unit spots, including USD=1.
        domestic_rates: Annual decimal short rates for every required currency.
        asset_prices: Positive native-currency total-return/adjusted price histories.
        asset_metadata: Asset-indexed currency, hedge_ratio and return_frequency columns.
        settings: Complete versioned estimation and coordinate configuration.
        provenance: Optional descriptive caller provenance; loaded bundles add source hashes.

    Loading is complete when validation succeeds; this object contains no fitted
    model. Fitting and persistence validate copies again, so subsequent edits to
    inspectable frames cannot silently bypass these checks. Date indices declare
    availability: publication lags/restatements cannot be inferred from prices.
    Missing observations are retained for the upstream numerical policies;
    infinities and columns unavailable by cutoff are rejected. Histories include
    warm-up before estimation_start. The cutoff need not itself be a trading date.
    """

    factor_levels: pd.DataFrame
    fx_spots: pd.DataFrame
    domestic_rates: pd.DataFrame
    asset_prices: pd.DataFrame
    asset_metadata: pd.DataFrame
    settings: BundleSettings
    provenance: dict | None = None

    def __post_init__(self) -> None:
        """Copy frames and validate ordering, coordinates, currencies and history coverage."""
        if not isinstance(self.settings, BundleSettings):
            raise TypeError("settings must be BundleSettings")
        for name in ("factor_levels", "fx_spots", "domestic_rates", "asset_prices"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"{name} must be a nonempty DataFrame")
            if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.index.tz is not None
                    or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
                raise ValueError(f"{name} dates must be timezone-naive, sorted and unique")
            if not frame.columns.is_unique or any(not isinstance(c, str) or not c.strip() for c in frame.columns):
                raise ValueError(f"{name} columns must have nonempty unique labels")
            values = frame.to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"{name} must contain finite numeric values or missing observations")
            if frame.loc[:self.settings.estimation_end].isna().all().any():
                raise ValueError(f"{name} has a column with no observations at the cutoff")
            if frame.index[0] >= self.settings.estimation_start:
                raise ValueError(f"{name} must contain history before estimation_start")
            object.__setattr__(self, name, frame.astype(float).copy())
        if tuple(self.factor_levels.columns) != self.settings.factor_names:
            raise ValueError("factor_levels columns must match ordered factor_names")
        for name in ("asset_prices", "fx_spots"):
            if (getattr(self, name) <= 0).any().any():
                raise ValueError(f"{name} must contain strictly positive levels")
        log_factors = [name for name, kind in self.settings.factor_return_types.items() if kind == "LOG"]
        if (self.factor_levels[log_factors] <= 0).any().any():
            raise ValueError("LOG factor levels must be strictly positive")
        metadata = self.asset_metadata.copy()
        if (not metadata.index.is_unique or set(metadata.index) != set(self.asset_prices.columns)
                or set(metadata.columns) != {"currency", "hedge_ratio", "return_frequency"}):
            raise ValueError("asset_metadata must match assets and contain exactly currency, hedge_ratio, return_frequency")
        metadata = metadata.reindex(self.asset_prices.columns)
        metadata["hedge_ratio"] = pd.to_numeric(metadata["hedge_ratio"], errors="raise")
        if not metadata["hedge_ratio"].between(0., 1.).all():
            raise ValueError("hedge_ratio must be finite and between zero and one")
        for frequency in metadata["return_frequency"].unique():
            _frequency(frequency, "return_frequency")
        if any(not isinstance(value, str) or not value.strip() for value in metadata["currency"]):
            raise ValueError("currency must be a nonempty string")
        currencies = set(metadata["currency"]) | {self.settings.reference_ccy, "USD"}
        if not currencies.issubset(self.fx_spots.columns) or not currencies.issubset(self.domestic_rates.columns):
            raise ValueError("FX spots/rates must cover USD and all asset/reference currencies")
        if not np.allclose(self.fx_spots["USD"].dropna(), 1., rtol=0, atol=1e-12):
            raise ValueError("USD anchor spots must equal one in USD_per_currency convention")
        object.__setattr__(self, "asset_metadata", metadata)
        object.__setattr__(self, "provenance", deepcopy(self.provenance or {}))


def _read_csv(payload: bytes, filename: str, *, dated: bool) -> pd.DataFrame:
    """Read a declared CSV without pandas silently mangling duplicate column labels."""
    text = payload.decode("utf-8-sig")
    header = next(csv.reader(StringIO(text)), [])
    if len(header) < 2 or len(set(header[1:])) != len(header[1:]):
        raise ValueError(f"{filename} has missing or duplicate column labels")
    frame = pd.read_csv(StringIO(text), index_col=0, dtype=None if dated else str, keep_default_na=dated)
    if dated:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    return frame


def load_bundle(directory: str | Path) -> FactorBundle:
    """Read and validate six supplied files, without fitting, fetching or writing.

    Args:
        directory: Existing six-file directory; relative names use YAML RESOURCE_PATH.

    Returns:
        FactorBundle: Owned histories and explicit settings, with source SHA-256 hashes.

    Raises:
        FileNotFoundError: If any required file is absent.
        ValueError: If schema, labels, units, chronology or configuration is invalid.
    """
    directory = resolve_resource_path(directory)
    missing = [name for name in _FILES.values() if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete bundle in {directory}: missing {missing}")
    payloads = {filename: (directory / filename).read_bytes() for filename in _FILES.values()}
    settings = BundleSettings.from_frame(_read_csv(payloads[_FILES["settings"]], _FILES["settings"], dated=False))
    frames = {name: _read_csv(payloads[filename], filename, dated=name != "asset_metadata")
              for name, filename in _FILES.items() if name != "settings"}
    return FactorBundle(**frames, settings=settings, provenance={
        "source_directory": str(directory),
        "source_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    })


def fit_bundle(bundle: FactorBundle, *, rolling: bool = False) -> FactorModelSnapshot | RollingFactorModel:
    """Fit supplied histories in memory, enforcing the configured information cutoff.

    Args:
        bundle: Validated inputs; a fresh copy is checked at the fit boundary.
        rolling: False returns one snapshot at estimation_end. True returns dated
            snapshots from estimation_start through estimation_end.

    Returns:
        FactorModelSnapshot | RollingFactorModel: Complete FM output, with currency,
        units, recipe, configuration and source provenance attached to each snapshot.

    The start date selects rolling output, never removes earlier warm-up history.
    All histories are truncated before QIS currency conversion. No fitted output
    or source file is persisted or replaced by this call.
    """
    bundle = replace(bundle)
    if type(rolling) is not bool:
        raise ValueError("rolling must be a boolean")
    settings = bundle.settings
    cutoff = settings.estimation_end
    fx = qis.FxRatesData(fx_spots=bundle.fx_spots.loc[:cutoff].copy(),
                        domestic_rates=bundle.domestic_rates.loc[:cutoff].copy())
    returns = fx.compute_fx_adjusted_returns(
        prices=bundle.asset_prices.loc[:cutoff].copy(),
        hedge_ratios=bundle.asset_metadata["hedge_ratio"],
        local_ccys=bundle.asset_metadata["currency"], reference_ccy=settings.reference_ccy,
        freq=bundle.asset_metadata["return_frequency"], is_log_returns=settings.is_log_returns,
        is_excess_returns=settings.is_excess_returns,
        zero_return_to_nan=settings.zero_return_to_nan, cash_rate_lag=settings.cash_rate_lag,
    )
    estimator = settings.make_estimator()
    arguments = dict(risk_factor_prices=bundle.factor_levels.loc[:cutoff].copy(),
                     asset_returns_dict=returns, assets=bundle.asset_prices.columns)
    if rolling:
        result = estimator.fit_rolling(**arguments, time_period=qis.TimePeriod(
            start=settings.estimation_start, end=cutoff))
    else:
        result = estimator.fit_current(**arguments, cutoff=cutoff)

    def annotate(snapshot: FactorModelSnapshot) -> FactorModelSnapshot:
        """Attach declared source/coordinate metadata without changing fitted values."""
        return replace(snapshot, provenance={
            **snapshot.provenance, "bundle_schema_version": settings.schema_version,
            "reference_ccy": settings.reference_ccy, "factor_units": dict(settings.factor_units),
            "factor_return_basis": settings.factor_return_basis, "recipe_id": settings.recipe_id,
            "bundle_settings": settings.to_frame()["value"].to_dict(),
            "asset_metadata": bundle.asset_metadata.to_dict(orient="index"),
            "input_sha256": {
                name: hashlib.sha256((settings.to_frame() if name == "settings" else getattr(bundle, name))
                                     .to_csv().encode("utf-8")).hexdigest()
                for name in _FILES
            },
            "bundle_source": deepcopy(bundle.provenance),
        })
    return RollingFactorModel({date: annotate(value) for date, value in result.snapshots.items()}) if rolling else annotate(result)


def save_bundle(bundle: FactorBundle, directory: str | Path) -> Path:
    """Persist validated input copies to a new directory; never replace existing output.

    Args:
        bundle: Inputs to serialize, checked again before writing.
        directory: New destination; relative names use YAML OUTPUT_PATH. Missing parents are created.

    Returns:
        Path: Absolute path containing the six versioned CSV files.

    Raises:
        FileExistsError: If the destination already exists, even if it is empty.

    This persists model inputs, not a fitted-result checkpoint. Interrupted writes
    may leave an incomplete new directory; the loader rejects incomplete bundles.
    """
    bundle = replace(bundle)
    directory = create_output_directory(directory)
    for name, filename in _FILES.items():
        frame = bundle.settings.to_frame() if name == "settings" else getattr(bundle, name)
        frame.to_csv(directory / filename)
    return directory
