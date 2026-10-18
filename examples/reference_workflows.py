"""Three explicit reference paths: synthetic, supplied bundle, public proxy.

All prices in the default example are synthetic USD total-return NAVs. Values
are monthly decimal log returns; covariance is annual. No acquisition, fitting
on load, or persistence occurs implicitly. Run with --help for optional actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradablefactors import (
    BundleSettings, FactorBundle, RiskFactors, fit_bundle, load_bundle, save_bundle,
)
from tradablefactors.cma import CmaDiagnosticInputs, compute_cma_diagnostics
from tradablefactors.reports.cma_diagnostics import render_cma_diagnostics


def synthetic_bundle() -> FactorBundle:
    """Return reproducible, explicitly labelled inputs; nothing is fitted or saved."""
    rng = np.random.default_rng(6102026)
    dates = pd.date_range("2014-01-31", periods=121, freq="ME")
    x = rng.normal([.004, .001], [.035, .018], (120, 2))
    b = np.array([[.9, .1], [.1, .9], [.6, .4], [.4, -.2]])
    y = x @ b.T + rng.normal(0., .013, (120, 4))
    levels = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(2), x.cumsum(axis=0)])),
                          index=dates, columns=["Equity", "Rates"])
    prices = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(4), y.cumsum(axis=0)])),
                          index=dates, columns=["Asset A", "Asset B", "Asset C", "Asset D"])
    metadata = pd.DataFrame({"currency": "USD", "hedge_ratio": 0., "return_frequency": "ME"},
                            index=prices.columns)
    return FactorBundle(
        factor_levels=levels, fx_spots=pd.DataFrame({"USD": 1.}, index=dates),
        domestic_rates=pd.DataFrame({"USD": 0.}, index=dates), asset_prices=prices,
        asset_metadata=metadata,
        settings=BundleSettings(reference_ccy="USD", factor_names=tuple(levels.columns),
                                estimation_start=dates[36], estimation_end=dates[-1],
                                recipe_id="synthetic-reference-v1"),
        provenance={"data_kind": "synthetic", "seed": 6102026})


def inspect_fit(bundle: FactorBundle) -> dict:
    """Fit supplied inputs and check covariance assembly independently of FL helpers."""
    snapshot = fit_bundle(bundle)
    betas = snapshot.data.y_betas
    factors = snapshot.data.x_covar
    residual = snapshot.residual_covariance
    covariance = snapshot.asset_covariance
    # Numerical reference only: explicit component sums, independent of get_y_covar.
    reference = np.einsum("ik,kl,jl->ij", betas, factors, betas) + residual.to_numpy()
    np.testing.assert_allclose(covariance, reference, rtol=1e-11, atol=1e-12)
    assert covariance.index.equals(betas.index) and covariance.columns.equals(betas.index)
    assert snapshot.cutoff == bundle.settings.estimation_end
    assert snapshot.data.residuals.index.max() <= snapshot.cutoff
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10

    # The renderer uses percentage/bp coordinates and diagonal GLS error geometry.
    # A valid model outside that report's scope still completes without a report.
    if (any(kind != "LOG" for kind in bundle.settings.factor_return_types.values())
            or snapshot.residual_type != "orthogonal" or (np.diag(residual) <= 0).any()):
        return {"bundle": bundle, "snapshot": snapshot, "audit": None,
                "report_status": "unavailable: illustrative CMA report requires LOG factors "
                                 "and positive orthogonal selected residual variances"}
    # Forecasts are supplied assumptions, distinct from the historical fit.
    premia = pd.Series(np.linspace(.05, .02, len(betas.columns)), index=betas.columns)
    forecast = betas @ premia + pd.Series(np.linspace(.006, -.004, len(betas)), index=betas.index)
    audit = compute_cma_diagnostics(CmaDiagnosticInputs(
        betas=betas, factor_covariance=factors,
        residual_variances=pd.Series(np.diag(residual), index=betas.index),
        excess_cma=forecast, factor_premia=premia,
        production_covariance=covariance,
        provenance={"reference_ccy": bundle.settings.reference_ccy, "as_of": str(snapshot.cutoff.date()),
                    "universe": f"{len(betas)} supplied assets", "model_name": bundle.settings.recipe_id,
                    "forecast_basis": "illustrative annual decimal excess returns"}))
    assert not audit.assets.empty and not audit.validation.empty
    return {"bundle": bundle, "snapshot": snapshot, "audit": audit, "report_status": "available"}


def persist_example(result: dict, directory: Path) -> Path:
    """Explicitly save inspectable tables and a PDF to a new caller-chosen directory.

    These CSVs are table exports, not a fitted-object checkpoint or reload format.
    The six input CSVs under inputs/ can be loaded separately without refitting.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_bundle(result["bundle"], directory / "inputs")
    snapshot, audit = result["snapshot"], result["audit"]
    snapshot.data.y_betas.to_csv(directory / "loadings.csv")
    snapshot.data.x_covar.to_csv(directory / "factor_covariance.csv")
    snapshot.residual_covariance.to_csv(directory / "residual_covariance.csv")
    snapshot.asset_covariance.to_csv(directory / "asset_covariance.csv")
    (directory / "result_metadata.json").write_text(json.dumps({
        "cutoff": str(snapshot.cutoff), "reference_ccy": result["bundle"].settings.reference_ccy,
        "covariance_units": "annual_asset_log_returns_squared", "residual_type": snapshot.residual_type,
        "factor_return_types": dict(result["bundle"].settings.factor_return_types),
        "factor_units": dict(result["bundle"].settings.factor_units),
        "report_status": result["report_status"], "provenance": snapshot.provenance,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    if audit is not None:
        for name in audit.__dataclass_fields__:
            table = getattr(audit, name)
            if isinstance(table, pd.DataFrame):
                table.to_csv(directory / f"{name}.csv")
        (directory / "audit_metadata.json").write_text(json.dumps({
            "summary": audit.summary, "provenance": audit.provenance,
        }, indent=2, default=str) + "\n", encoding="utf-8")
        render_cma_diagnostics(audit, directory / "cma_report.pdf", pages=2)
    return directory


class ProxyFactors(RiskFactors):
    """Two deliberately limited long-only proxy definitions."""

    EQUITY = "Equity"
    RATES = "Rates"


def public_proxy_bundle(prices: pd.DataFrame | None = None) -> FactorBundle:
    """Construct a two-ETF proxy from supplied USD adjusted prices.

    None uses synthetic histories labelled SPY/TLT, not downloaded ETF evidence.
    Equity=SPY and Rates=TLT are simple long-only proxy definitions; they do not
    reproduce the proprietary MATF universe, hedges, carry or calibration.
    """
    from tradablefactors import Factor, FactorFamily, VolTargetPolicy, build_factor_panel
    from tradablefactors.spec import FactorModelSpec

    base = synthetic_bundle()
    synthetic_prices = prices is None
    if prices is None:
        prices = base.factor_levels.rename(columns={"Equity": "SPY", "Rates": "TLT"})
    prices = prices.loc[:, ["SPY", "TLT"]].copy()
    factors = [Factor(name="Equity", family=FactorFamily("Equity", vol_target=None),
                      long_side_tickers=("SPY",), long_side_weights=(1.,),
                      vol_target_policy=VolTargetPolicy.NONE),
               Factor(name="Rates", family=FactorFamily("Rates", vol_target=None),
                      long_side_tickers=("TLT",), long_side_weights=(1.,),
                      vol_target_policy=VolTargetPolicy.NONE)]
    spec = FactorModelSpec(name="Public ETF proxy", factors=ProxyFactors,
                           baskets=dict(zip(ProxyFactors, factors)), file_name="etf_proxy")
    levels = build_factor_panel(spec, prices, returns_freq="ME", rebalancing_freq="QE", vol_span=None)
    # Four illustrative targets remain synthetic even when the proxy prices are acquired.
    # Use supplied proxy date overlap; do not imply that these are observed asset returns.
    assets = base.asset_prices.reindex(levels.index).dropna()
    if len(assets) < 24:
        raise ValueError("The teaching asset panel requires at least 24 common dates in 2014–2024")
    levels = levels.loc[assets.index]
    return FactorBundle(
        factor_levels=levels, fx_spots=pd.DataFrame({"USD": 1.}, index=levels.index),
        domestic_rates=pd.DataFrame({"USD": 0.}, index=levels.index), asset_prices=assets,
        asset_metadata=base.asset_metadata,
        settings=BundleSettings(reference_ccy="USD", factor_names=tuple(levels.columns),
                                estimation_start=levels.index[12], estimation_end=levels.index[-1],
                                recipe_id="public-etf-proxy-v1"),
        provenance={"proxy_prices": "synthetic" if synthetic_prices else "caller_supplied_adjusted_closes",
                    "target_assets": "synthetic", "definitions": {"Equity": "long SPY", "Rates": "long TLT"}})


def run_local_test(case: int = 1) -> dict:
    """Run synthetic fit (1) or offline proxy construction/fit (2), returning objects."""
    if case not in (1, 2):
        raise ValueError("case must be 1 (synthetic) or 2 (synthetic ETF proxy)")
    result = inspect_fit(synthetic_bundle() if case == 1 else public_proxy_bundle())
    print(result["snapshot"].asset_covariance)
    return result


def acquire_proxy_prices() -> pd.DataFrame:
    """Explicit optional acquisition; adjusted Yahoo closes, never required offline.

    The fixed date interval supports the teaching panel. Provider revisions,
    licensing and dividend/adjustment quality remain the caller's responsibility.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Optional acquisition requires installing yfinance separately") from exc
    raw = yf.download(tickers=["SPY", "TLT"], start="2014-01-01", end="2024-02-01",
                      auto_adjust=True, ignore_tz=True, progress=False, threads=False,
                      multi_level_index=True)
    if raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise ValueError("No adjusted closes returned; supply your own validated CSV")
    prices = raw["Close"].loc[:, ["SPY", "TLT"]].resample("ME").last()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Proxy adjusted closes must be complete and positive")
    return prices


def main() -> None:
    """Load only by default; fitting and writing each require their own option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Caller-supplied schema-v1 six-file directory")
    parser.add_argument("--proxy-prices", type=Path, help="Already acquired USD adjusted SPY/TLT CSV")
    parser.add_argument("--acquire-proxy", action="store_true", help="Explicit optional Yahoo acquisition")
    parser.add_argument("--fit", action="store_true", help="Explicitly fit in memory")
    parser.add_argument("--output", type=Path, help="New output directory; requires --fit")
    args = parser.parse_args()
    if sum(bool(x) for x in (args.bundle, args.proxy_prices, args.acquire_proxy)) > 1:
        parser.error("choose --bundle, --proxy-prices or --acquire-proxy")
    if args.output and not args.fit:
        parser.error("--output requires --fit")
    bundle = (public_proxy_bundle(acquire_proxy_prices()) if args.acquire_proxy else
        load_bundle(args.bundle) if args.bundle else public_proxy_bundle(
        pd.read_csv(args.proxy_prices, index_col=0, parse_dates=True))
        if args.proxy_prices else synthetic_bundle())
    print(f"Loaded {len(bundle.factor_levels)} factor observations; recipe={bundle.settings.recipe_id}")
    if args.fit:
        result = inspect_fit(bundle)
        print(result["snapshot"].asset_covariance)
        print(result["report_status"])
        if args.output:
            persist_example(result, args.output)


if __name__ == "__main__":
    main()
