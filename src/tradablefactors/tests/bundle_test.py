"""Installed six-file bundle contracts, with independent return/covariance checks."""
from dataclasses import replace
import hashlib
import socket

import numpy as np
import pandas as pd
import pytest
from qis.perfstats.config import ReturnTypes

from tradablefactors import BundleSettings, FactorBundle, fit_bundle, load_bundle, save_bundle


def example_bundle():
    """Return deterministic supplied monthly histories with a nontrivial residual block."""
    rng = np.random.default_rng(61726)
    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    factor_returns = rng.normal([.004, .001], [.035, .014], size=(len(dates), 2))
    residual = rng.normal(0, .009, (len(dates), 2))
    asset_returns = factor_returns @ np.array([[.8, -.2], [.3, .7]]) + residual
    return FactorBundle(
        factor_levels=pd.DataFrame(np.exp(np.cumsum(factor_returns, axis=0)), index=dates, columns=["Equity", "Rates"]),
        asset_prices=pd.DataFrame(100 * np.exp(np.cumsum(asset_returns, axis=0)), index=dates, columns=["A", "B"]),
        fx_spots=pd.DataFrame({"USD": 1.}, index=dates),
        domestic_rates=pd.DataFrame({"USD": 0.}, index=dates),
        asset_metadata=pd.DataFrame({"currency": "USD", "hedge_ratio": 0., "return_frequency": "ME"}, index=["A", "B"]),
        settings=BundleSettings(reference_ccy="USD", factor_names=("Equity", "Rates"),
                                estimation_start="2020-12-31", estimation_end=dates[-1], recipe_id="synthetic-v1",
                                rebalancing_freq="YE", loss_normalization="weight_sum"),
    )


def _hashes(directory):
    """Read content hashes so load/fit mutation assertions cover all files."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}


def test_load_is_offline_read_only_and_does_not_fit(tmp_path, monkeypatch):
    """Loading completes a bundle without estimator execution or network access."""
    directory = save_bundle(example_bundle(), tmp_path / "input")
    before = _hashes(directory)

    def forbidden(*args, **kwargs):
        """Fail if loading tries to fit or access the network."""
        raise AssertionError("load must only read")

    from tradablefactors import FactorCovarEstimator
    monkeypatch.setattr(FactorCovarEstimator, "fit_current", forbidden)
    monkeypatch.setattr(FactorCovarEstimator, "fit_rolling", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    loaded = load_bundle(directory)
    assert loaded.settings.schema_version == "1"
    assert len(loaded.provenance["source_sha256"]) == 6
    assert _hashes(directory) == before
    pd.testing.assert_frame_equal(loaded.factor_levels, example_bundle().factor_levels, check_freq=False)


def test_settings_roundtrip_and_configuration_handoff():
    """Nondefault cadence, penalty, residual and coordinate choices reach the estimator."""
    settings = replace(example_bundle().settings, residual_type="empirical", residual_covar_freq="ME",
                       residual_covar_span=18., residual_corr_weight=.4, residual_var_weight=.8,
                       span_freq_dict={"ME": 24}, reg_lambda_freq_dict={"ME": 2.e-5},
                       factor_return_types={"Equity": "LOG", "Rates": "DIFFERENCE"},
                       factor_units={"Equity": "dimensionless", "Rates": "index_points"})
    reconstructed = BundleSettings.from_frame(settings.to_frame())
    pd.testing.assert_frame_equal(settings.to_frame(), reconstructed.to_frame())
    estimator = reconstructed.make_estimator()
    assert estimator.reg_lambda_freq_dict == {"ME": 2.e-5}
    assert estimator.lasso_model.span_freq_dict == {"ME": 24}
    assert estimator.lasso_model.loss_normalization == "weight_sum"
    assert estimator.residual_type == "empirical"
    assert estimator.residual_corr_weight == .4
    assert estimator.residual_var_weight == .8
    assert estimator.factor_return_types["Rates"] is ReturnTypes.DIFFERENCE


@pytest.mark.parametrize("change,match", [
    ({"schema_version": "2"}, "schema_version"),
    ({"is_log_returns": "true"}, "boolean"),
    ({"is_log_returns": False}, "is_log_returns"),
    ({"factor_covar_span": 2.5}, "positive integer"),
    ({"factor_covar_span": 0}, "positive integer"),
    ({"warmup_period": -1}, "positive integer"),
    ({"reg_lambda": float("nan")}, "reg_lambda"),
    ({"loss_normalization": "invented"}, "loss_normalization"),
    ({"factor_returns_freq": "nonsense"}, "frequency"),
    ({"factor_returns_freq": "-1ME"}, "frequency"),
    ({"lasso_model_type": "invented"}, "lasso_model_type"),
    ({"estimation_end": "2010-01-01"}, "estimation_start"),
    ({"estimation_end": "NaT"}, "estimation_end"),
    ({"rates_unit": "percent"}, "rates_unit"),
    ({"fx_spot_convention": "currency_per_USD"}, "fx_spot_convention"),
    ({"covariance_units": "monthly"}, "covariance_units"),
    ({"factor_return_basis": "excess"}, "factor_return_basis"),
    ({"factor_names": ("Equity", "Equity")}, "factor_names"),
    ({"factor_units": {"Equity": "points", "Rates": "dimensionless"}}, "LOG requires"),
    ({"factor_return_types": {"Equity": "LOG", "Rates": "SIMPLE"}}, "factor_return_types"),
    ({"factor_return_types": {"Equity": "LOG", "Rates": "DIFFERENCE"}}, "LOG requires"),
])
def test_bad_settings_fail_before_fit(change, match):
    """Invalid units/configuration must fail while constructing the loadable object."""
    with pytest.raises(ValueError, match=match):
        replace(example_bundle().settings, **change)


@pytest.mark.parametrize("defect,match", [
    ("missing", "incomplete"), ("legacy", "schema_version"), ("duplicate_setting", "unique setting"),
    ("unknown_setting", "unknown"), ("fractional_integer", "positive integer"),
    ("bad_bool", "boolean"), ("duplicate_header", "duplicate column"),
    ("factor_order", "ordered factor_names"), ("duplicate_dates", "sorted and unique"),
    ("unsorted_dates", "sorted and unique"), ("nonfinite", "finite numeric"),
    ("negative_nav", "LOG factor"), ("negative_asset", "positive levels"),
    ("usd_anchor", "USD anchor"), ("missing_currency", "currencies"),
    ("duplicate_asset", "asset_metadata"), ("hedge_ratio", "hedge_ratio"),
    ("frequency", "frequency"), ("missing_history", "history before estimation_start"),
])
def test_invalid_supplied_files_fail_closed(tmp_path, defect, match):
    """Reject corrupt CSVs instead of sorting, filling, changing units or accepting legacy data."""
    directory = save_bundle(example_bundle(), tmp_path / "bundle")
    filename = "risk_model_settings.csv"
    if defect == "missing":
        (directory / "asset_prices.csv").unlink()
    elif defect == "duplicate_header":
        path = directory / "futures_risk_factors.csv"
        path.write_text(path.read_text().replace("Equity,Rates", "Equity,Equity"))
    elif defect in {"legacy", "duplicate_setting", "unknown_setting", "fractional_integer", "bad_bool"}:
        frame = pd.read_csv(directory / filename, index_col=0, dtype=str)
        if defect == "legacy":
            frame = frame.drop("schema_version")
        elif defect == "duplicate_setting":
            frame = pd.concat([frame, frame.loc[["solver"]]])
        elif defect == "unknown_setting":
            frame.loc["typo"] = "1"
        elif defect == "fractional_integer":
            frame.loc["beta_span"] = "2.5"
        else:
            frame.loc["demean"] = "maybe"
        frame.to_csv(directory / filename)
    else:
        filename = {
            "factor_order": "futures_risk_factors.csv", "negative_nav": "futures_risk_factors.csv",
            "usd_anchor": "fx_hedging_data_fx_spots.csv", "missing_currency": "asset_metadata.csv",
            "duplicate_asset": "asset_metadata.csv", "hedge_ratio": "asset_metadata.csv",
            "frequency": "asset_metadata.csv",
        }.get(defect, "asset_prices.csv")
        frame = pd.read_csv(directory / filename, index_col=0)
        if defect == "factor_order":
            frame = frame.iloc[:, ::-1]
        elif defect in {"duplicate_dates", "duplicate_asset"}:
            frame = pd.concat([frame, frame.tail(1)])
        elif defect == "unsorted_dates":
            frame = frame.iloc[::-1]
        elif defect == "nonfinite":
            frame.iloc[0, 0] = np.inf
        elif defect in {"negative_nav", "negative_asset"}:
            frame.iloc[0, 0] = -1.
        elif defect == "usd_anchor":
            frame["USD"] = 1.1
        elif defect == "missing_currency":
            frame["currency"] = "EUR"
        elif defect == "hedge_ratio":
            frame["hedge_ratio"] = 1.1
        elif defect == "frequency":
            frame["return_frequency"] = "invalid"
        else:
            frame = frame.tail(1)
        frame.to_csv(directory / filename)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_bundle(directory)


def test_fit_matches_direct_return_reference_and_covariance_identity(tmp_path):
    """Compare QIS conversion handoff to direct USD log differences and independent B F B' + R."""
    original = example_bundle()
    directory = save_bundle(original, tmp_path / "bundle")
    before = _hashes(directory)
    loaded = load_bundle(directory)
    actual = fit_bundle(loaded)
    direct_returns = np.log(original.asset_prices).diff()
    reference = original.settings.make_estimator().fit_current(
        original.factor_levels, {"ME": direct_returns}, cutoff=original.settings.estimation_end,
        assets=original.asset_prices.columns)
    pd.testing.assert_frame_equal(actual.data.y_betas, reference.data.y_betas, rtol=1e-11, atol=1e-12)
    pd.testing.assert_frame_equal(actual.asset_covariance, reference.asset_covariance, rtol=1e-11, atol=1e-12)
    b, f, r = actual.data.y_betas.to_numpy(), actual.data.x_covar.to_numpy(), actual.residual_covariance.to_numpy()
    independent = np.array([[sum(b[i, k] * f[k, ell] * b[j, ell]
                                 for k in range(2) for ell in range(2)) + r[i, j]
                             for j in range(2)] for i in range(2)])
    np.testing.assert_allclose(actual.asset_covariance, independent, rtol=1e-12, atol=1e-14)
    factor_returns = np.log(original.factor_levels).diff().iloc[1:]
    demeaned = (factor_returns - factor_returns.ewm(span=original.settings.factor_covar_span, adjust=False).mean()).iloc[1:]
    decay = 1. - 2. / (original.settings.factor_covar_span + 1.)
    weights = (1. - decay) * decay ** np.arange(len(demeaned) - 1, -1, -1)
    annual_reference = 12. * np.einsum("t,ti,tj->ij", weights, demeaned, demeaned)
    np.testing.assert_allclose(actual.data.x_covar, annual_reference, rtol=1e-12, atol=1e-14)
    assert actual.cutoff == original.settings.estimation_end
    assert actual.provenance["reference_ccy"] == "USD"
    assert actual.provenance["factor_covariance_units"] == "annual_native_coordinates"
    assert actual.provenance["bundle_source"]["source_sha256"] == before
    assert _hashes(directory) == before


def test_cutoff_future_invariance_and_rolling_outputs():
    """Future factor/asset/FX/rate mutations cannot affect any fitted snapshot."""
    original = example_bundle()
    early = replace(original, settings=replace(original.settings, estimation_end=pd.Timestamp("2021-12-31")))
    changed = replace(early)
    for frame in (changed.factor_levels, changed.asset_prices):
        frame.loc[frame.index > early.settings.estimation_end] *= 7.
    changed.domestic_rates.loc[changed.domestic_rates.index > early.settings.estimation_end, "USD"] = .3
    first = fit_bundle(early)
    second = fit_bundle(changed)
    pd.testing.assert_frame_equal(first.data.y_betas, second.data.y_betas)
    pd.testing.assert_frame_equal(first.asset_covariance, second.asset_covariance)
    rolling = fit_bundle(changed, rolling=True)
    assert list(rolling.snapshots) == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]
    pd.testing.assert_frame_equal(rolling.snapshots[first.cutoff].asset_covariance, first.asset_covariance)


def test_explicit_persistence_never_replaces_and_revalidates(tmp_path):
    """An edited result must pass validation again and existing files are preserved."""
    bundle = example_bundle()
    directory = save_bundle(bundle, tmp_path / "new")
    before = _hashes(directory)
    with pytest.raises(FileExistsError):
        save_bundle(bundle, directory)
    assert _hashes(directory) == before
    bundle.asset_prices.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="positive levels"):
        save_bundle(bundle, tmp_path / "invalid")
    with pytest.raises(ValueError, match="positive levels"):
        fit_bundle(bundle)
    assert not (tmp_path / "invalid").exists()


def test_missing_observations_preserved_and_fitted(tmp_path):
    """Different inception dates and missing samples remain missing through CSV load."""
    bundle = example_bundle()
    bundle.asset_prices.loc[bundle.asset_prices.index[:12], "B"] = np.nan
    bundle.factor_levels.iloc[8, 1] = np.nan
    directory = save_bundle(bundle, tmp_path / "missing-history")
    loaded = load_bundle(directory)
    np.testing.assert_array_equal(loaded.asset_prices.isna(), bundle.asset_prices.isna())
    np.testing.assert_array_equal(loaded.factor_levels.isna(), bundle.factor_levels.isna())
    result = fit_bundle(loaded)
    assert list(result.data.y_betas.index) == ["A", "B"]
    assert np.isfinite(result.asset_covariance.to_numpy()).all()
    loaded.asset_prices["B"] = np.nan
    with pytest.raises(ValueError, match="no observations at the cutoff"):
        fit_bundle(loaded)


def test_native_difference_units_survive_load_and_fit(tmp_path):
    """Signed native levels retain their scaling instead of becoming log NAVs."""
    original = example_bundle()
    levels = original.factor_levels.copy()
    levels["Rates"] = -5. + 10. * np.log(levels["Rates"])
    native = replace(original, factor_levels=levels, settings=replace(
        original.settings, factor_return_types={"Equity": "LOG", "Rates": "DIFFERENCE"},
        factor_units={"Equity": "dimensionless", "Rates": "index_points"}))
    loaded = load_bundle(save_bundle(native, tmp_path / "native"))
    result = fit_bundle(loaded)
    reference = fit_bundle(original)
    scale = np.array([1., 10.])
    np.testing.assert_allclose(result.data.x_covar, reference.data.x_covar.to_numpy() * scale[:, None] * scale[None, :],
                               rtol=1e-11, atol=1e-13)
    assert result.factor_return_types["Rates"] is ReturnTypes.DIFFERENCE
    assert result.provenance["factor_units"]["Rates"] == "index_points"


def test_currency_hedging_handoff_matches_endpoint_reference():
    """Reference-currency returns agree with endpoint wealth and contracted forward arithmetic."""
    original = example_bundle()
    dates = original.asset_prices.index
    spots = pd.DataFrame({"USD": 1., "CHF": np.exp(.03 * np.sin(np.arange(len(dates)) / 7.))}, index=dates)
    rates = pd.DataFrame({"USD": .02, "CHF": .01}, index=dates)
    metadata = original.asset_metadata.copy()
    metadata.loc["B", "hedge_ratio"] = 1.
    bundle = replace(original, fx_spots=spots, domestic_rates=rates, asset_metadata=metadata,
                     settings=replace(original.settings, reference_ccy="CHF"))
    gross_prices = original.asset_prices / original.asset_prices.shift()
    cross_spot = spots.USD / spots.CHF
    gross_fx = cross_spot / cross_spot.shift()
    forward_ratio = (1. + rates.CHF.shift() / 12.) / (1. + rates.USD.shift() / 12.)
    reference_returns = pd.DataFrame({
        asset: np.log(gross_prices[asset] * gross_fx + metadata.loc[asset, "hedge_ratio"] * (forward_ratio - gross_fx))
        for asset in metadata.index})
    actual = fit_bundle(bundle)
    reference = bundle.settings.make_estimator().fit_current(
        bundle.factor_levels, {"ME": reference_returns}, cutoff=bundle.settings.estimation_end,
        assets=bundle.asset_prices.columns)
    pd.testing.assert_frame_equal(actual.data.y_betas, reference.data.y_betas, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(actual.asset_covariance, reference.asset_covariance, rtol=1e-10, atol=1e-12)


def test_nontrading_cutoff_does_not_require_future_observation():
    """An information cutoff after the last observation is valid and transparently dated."""
    original = example_bundle()
    later_cutoff = original.settings.estimation_end + pd.Timedelta(days=1)
    result = fit_bundle(replace(original, settings=replace(original.settings, estimation_end=later_cutoff)))
    reference = fit_bundle(original)
    assert result.cutoff == later_cutoff
    assert result.provenance["factor_observation_date"] == original.factor_levels.index[-1]
    pd.testing.assert_frame_equal(result.asset_covariance, reference.asset_covariance)
