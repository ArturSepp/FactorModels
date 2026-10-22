"""Three explicit reference paths: synthetic, supplied bundle, public proxy.

All prices in the default example are synthetic USD total-return NAVs with zero
cash. Fitted risk uses annualized log-return moments; illustrative forecasts use
annual arithmetic excess returns through an explicit second-order bridge.
No acquisition, fitting on load, or persistence occurs implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from factormodels.utils.local_path import create_output_directory, resolve_resource_path

import numpy as np
import pandas as pd

from factormodels import (
    BundleSettings, CmaDiagnosticInputs, FactorBundle, FactorModelSnapshot, RiskFactors,
    annual_arithmetic_from_log, compute_cma_diagnostics, fit_bundle,
    infer_factor_premia, load_bundle, save_bundle, synthetic_bundle,
)


def inspect_fit(bundle: FactorBundle) -> dict:
    """Fit supplied inputs and check covariance assembly independently of FL helpers."""
    snapshot = fit_bundle(bundle)
    betas = snapshot.loadings
    factors = snapshot.factor_covariance
    residual = snapshot.residual_covariance
    covariance = snapshot.asset_covariance
    # Numerical reference only: explicit component sums, independent of get_y_covar.
    reference = np.einsum("ik,kl,jl->ij", betas, factors, betas) + residual.to_numpy()
    np.testing.assert_allclose(covariance, reference, rtol=1e-11, atol=1e-12)
    assert covariance.index.equals(betas.index) and covariance.columns.equals(betas.index)
    assert snapshot.cutoff == bundle.settings.estimation_end
    assert snapshot.data.residuals.index.max() <= snapshot.cutoff
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10

    return {"bundle": bundle, "snapshot": snapshot, **audit_snapshot(bundle, snapshot)}


def audit_snapshot(bundle: FactorBundle, snapshot: FactorModelSnapshot) -> dict:
    """Connect a frozen fit to illustrative arithmetic forecasts without refitting.

    All factor forecasts are explicitly assumed to be 4% annual arithmetic excess
    returns. Historical annual log alpha uses an EWMA span of 36 observations.
    The bridge reconciles asset and factor means, including their different
    variance corrections. These teaching assumptions are not estimated CMAs.
    """
    betas, factors = snapshot.loadings, snapshot.factor_covariance
    residual, covariance = snapshot.residual_covariance, snapshot.asset_covariance
    unavailable = None
    if any(kind.name != "LOG" for kind in snapshot.factor_return_types.values()):
        unavailable = "native difference factors require explicit forecast units"
    elif not (bundle.domestic_rates.loc[:snapshot.cutoff].to_numpy() == 0.).all():
        unavailable = "nonzero cash requires explicit total-to-excess forecast treatment"
    elif not np.allclose(residual, np.diag(np.diag(residual)), rtol=0., atol=1e-14):
        unavailable = "correlated residual risk is retained; this report requires a diagonal policy"
    elif np.any(np.diag(residual) <= 0.):
        unavailable = "GLS requires strictly positive residual variances"
    if unavailable is not None:
        return {"audit": None, "report_status": f"unavailable: {unavailable}"}

    factor_arithmetic = pd.Series(.04, index=betas.columns, name="assumed_arithmetic_premium")
    factor_log = factor_arithmetic-.5*pd.Series(np.diag(factors), index=betas.columns)
    # FactorLasso estimates the EWMA from the producer's already annualized residuals.
    alpha_log = snapshot.data.estimate_alpha(alpha_span=36)
    asset_log = betas @ factor_log + alpha_log
    asset_arithmetic = annual_arithmetic_from_log(
        asset_log, pd.Series(np.diag(covariance), index=betas.index), annualization_factor=1.)
    alpha_arithmetic = asset_arithmetic-betas @ factor_arithmetic
    # A declared asset-forecast disagreement makes the GLS residual visible.
    forecast_tilt = pd.Series(np.linspace(-.0025, .0025, len(betas)), index=betas.index)
    total_forecast = asset_arithmetic+forecast_tilt
    premia = infer_factor_premia(
        betas, residual, total_forecast, cash_rate=0., alpha=alpha_arithmetic, alpha_weight=1.)
    inputs = CmaDiagnosticInputs(
        betas=betas, residual_variances=pd.Series(np.diag(residual), index=betas.index),
        factor_covariance=factors, production_covariance=covariance,
        excess_cma=total_forecast, factor_premia=premia.premia,
        admitted_alpha=alpha_arithmetic,
        channels=pd.DataFrame({"admitted_alpha": alpha_arithmetic,
                               "forecast_residual": premia.assets.residual}),
        provenance={"data_kind": "illustrative_forecasts_on_supplied_fit",
                    "reference_ccy": bundle.settings.reference_ccy,
                    "as_of": str(snapshot.cutoff.date()),
                    "model_name": "Annual log-risk approximation",
                    "universe": f"{len(betas)} assets / {len(betas.columns)} factors",
                    "risk_basis": "annualized_log_return_covariance",
                    "forecast_basis": "Annual arithmetic excess returns",
                    "forecast_units": "annual_arithmetic_excess_decimal",
                    "mean_risk_convention": "arithmetic forecasts with annualized log-return risk; "
                                            "not an exact annual compounded-return distribution",
                    "moment_bridge": "A * (log_mean + 0.5 * log_variance)",
                    "bridge_annualization_factor": 1., "alpha_span": 36,
                    "log_alpha_units": "already_annualized_decimal",
                    "factor_premium_assumptions": factor_arithmetic.to_dict(),
                    "forecast_tilt": forecast_tilt.to_dict(),
                    "cash_rate": 0., "calibrated_alpha_weight": 1.,
                    "residual_type": snapshot.residual_type,
                    "residual_var_weight": snapshot.residual_var_weight,
                    "audit_target": "full arithmetic excess forecast; admission remains a channel"})
    audit = compute_cma_diagnostics(inputs)
    assert audit.summary["accounting_status"] == "pass"
    forecasts = pd.DataFrame({"model_log_mean": asset_log, "model_arithmetic_mean": asset_arithmetic,
                              "historical_log_alpha": alpha_log, "credited_alpha": alpha_arithmetic,
                              "forecast_tilt": forecast_tilt, "provider_total": total_forecast,
                              "cash": 0., "gls_target": total_forecast-alpha_arithmetic,
                              "audit_excess_target": total_forecast})
    assumptions = pd.DataFrame({"arithmetic_premium": factor_arithmetic, "log_mean": factor_log})
    return {"audit": audit, "inputs": inputs, "premia": premia, "forecasts": forecasts,
            "factor_assumptions": assumptions,
            "report_status": "available: arithmetic forecast approximation with annualized log-return risk"}


def persist_example(result: dict, directory: Path) -> Path:
    """Save fitted tables and any eligible arithmetic audit on explicit request.

    These CSVs are table exports, not a fitted-object checkpoint or reload format.
    The six input CSVs under inputs/ can be loaded separately without refitting.
    """
    directory = create_output_directory(directory)
    save_bundle(result["bundle"], directory / "inputs")
    snapshot = result["snapshot"]
    snapshot.loadings.to_csv(directory / "loadings.csv")
    snapshot.factor_covariance.to_csv(directory / "factor_covariance.csv")
    snapshot.residual_covariance.to_csv(directory / "residual_covariance.csv")
    snapshot.asset_covariance.to_csv(directory / "asset_covariance.csv")
    (directory / "result_metadata.json").write_text(json.dumps({
        "cutoff": str(snapshot.cutoff), "reference_ccy": result["bundle"].settings.reference_ccy,
        "covariance_units": "annual_asset_log_returns_squared", "residual_type": snapshot.residual_type,
        "factor_return_types": dict(result["bundle"].settings.factor_return_types),
        "factor_units": dict(result["bundle"].settings.factor_units),
        "report_status": result["report_status"], "provenance": snapshot.provenance,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    if result["audit"] is not None:
        from factormodels.reports.cma_diagnostics import render_cma_diagnostics
        audit, inputs, premia = result["audit"], result["inputs"], result["premia"]
        for name in ("forecasts", "factor_assumptions"):
            result[name].to_csv(directory / f"{name}.csv")
        inputs.channels.to_csv(directory / "construction_channels.csv")
        for name in audit.__dataclass_fields__:
            table = getattr(audit, name)
            if isinstance(table, pd.DataFrame):
                table.to_csv(directory / f"{name}.csv")
        for name in ("assets", "factors", "covariance", "mimicking_weights"):
            getattr(premia, name).to_csv(directory / f"gls_alpha_1_{name}.csv")
        (directory / "audit_metadata.json").write_text(json.dumps({
            "summary": audit.summary, "provenance": audit.provenance, "gls_summary": premia.summary,
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
    from factormodels import Factor, FactorFamily, VolTargetPolicy, build_factor_panel
    from factormodels.spec import FactorModelSpec

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
        pd.read_csv(resolve_resource_path(args.proxy_prices), index_col=0, parse_dates=True))
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
