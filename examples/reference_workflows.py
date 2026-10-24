"""Three explicit reference paths: synthetic, supplied prices, public proxy.

All prices in the default example are synthetic USD total-return NAVs with zero
cash. Fitted risk uses annualized log-return moments; illustrative forecasts use
annual arithmetic excess returns through an explicit second-order bridge.
No acquisition, fitting on load, or persistence occurs implicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from factormodels.utils.local_path import create_output_directory, resolve_resource_path

import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType

from factormodels import (
    CmaDiagnosticInputs, FactorCovarEstimator, FactorModelSnapshot, RiskFactors,
    annual_arithmetic_from_log, compute_cma_diagnostics, infer_factor_premia,
)


def synthetic_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent synthetic factor NAVs and asset total-return prices.

    The fixed seed produces 121 monthly USD observations from January 2014 through
    January 2024 for two factors and four assets. These are illustrative prices,
    not market data. Callers prepare log returns and configure their estimator.
    No acquisition, fitting, rendering or persistence occurs.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Factor NAVs, then asset prices.
    """
    rng = np.random.default_rng(6102026)
    dates = pd.date_range("2014-01-31", periods=121, freq="ME")
    x = rng.normal([.004, .001], [.035, .018], (120, 2))
    b = np.array([[.9, .1], [.1, .9], [.6, .4], [.4, -.2]])
    y = x @ b.T + rng.normal(0., .013, (120, 4))
    levels = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(2), x.cumsum(axis=0)])),
                          index=dates, columns=["Equity", "Rates"])
    prices = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(4), y.cumsum(axis=0)])),
                          index=dates, columns=["Asset A", "Asset B", "Asset C", "Asset D"])
    return levels, prices


def inspect_fit(
    factor_prices: pd.DataFrame, asset_prices: pd.DataFrame, *,
    estimator: FactorCovarEstimator | None = None, cutoff=None,
    cash_rate: float = 0., factor_units: dict[str, str] | None = None,
) -> dict:
    """Fit supplied USD total-return prices directly and inspect the completed model.

    Inputs are monthly histories already in one currency; currency conversion and
    hedging belong to the caller. Cash is only a forecast-audit assumption here.
    Native factor levels require their units and an explicit estimator coordinate map.
    """
    cutoff = pd.Timestamp(factor_prices.index[-1] if cutoff is None else cutoff)
    if estimator is None:
        estimator = FactorCovarEstimator(
            lasso_model=LassoModel(
                model_type=LassoModelType.LASSO, reg_lambda=1.e-5, span=36,
                warmup_period=6, demean=True, solver="CLARABEL", loss_normalization="sample",
            ),
            factor_returns_freq="ME", factor_covar_span=36,
            rebalancing_freq="QE", residual_type="orthogonal",
        )
    # Keep warmup and the first missing return; exact zero returns remain missing.
    returns = qis.to_returns(
        asset_prices.loc[:cutoff], freq="ME", is_log_returns=True,
        drop_first=False, ffill_nans=False,
    ).replace(0., np.nan)
    snapshot = estimator.fit_current(
        risk_factor_prices=factor_prices, asset_returns_dict={"ME": returns},
        assets=list(asset_prices.columns), cutoff=cutoff,
    )
    if factor_units is None:
        if any(kind.name != "LOG" for kind in snapshot.factor_return_types.values()):
            raise ValueError("Native difference factors require explicit factor_units")
        factor_units = {name: "dimensionless" for name in factor_prices.columns}
    snapshot = replace(snapshot, provenance={
        **snapshot.provenance, "reference_ccy": "USD", "factor_units": dict(factor_units),
        "factor_return_basis": "total",
    })
    betas = snapshot.loadings
    factors = snapshot.factor_covariance
    residual = snapshot.residual_covariance
    covariance = snapshot.asset_covariance
    # Numerical reference only: explicit component sums, independent of get_y_covar.
    reference = np.einsum("ik,kl,jl->ij", betas, factors, betas) + residual.to_numpy()
    np.testing.assert_allclose(covariance, reference, rtol=1e-11, atol=1e-12)
    assert covariance.index.equals(betas.index) and covariance.columns.equals(betas.index)
    assert snapshot.cutoff == cutoff
    assert snapshot.data.residuals.index.max() <= snapshot.cutoff
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10

    return {"snapshot": snapshot, **audit_snapshot(snapshot, cash_rate=cash_rate)}


def audit_snapshot(snapshot: FactorModelSnapshot, *, cash_rate: float = 0.) -> dict:
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
    elif cash_rate != 0.:
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
                    "reference_ccy": snapshot.provenance["reference_ccy"],
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
    Retain source histories and estimator configuration separately when needed.
    """
    directory = create_output_directory(directory)
    snapshot = result["snapshot"]
    snapshot.loadings.to_csv(directory / "loadings.csv")
    snapshot.factor_covariance.to_csv(directory / "factor_covariance.csv")
    snapshot.residual_covariance.to_csv(directory / "residual_covariance.csv")
    snapshot.asset_covariance.to_csv(directory / "asset_covariance.csv")
    (directory / "result_metadata.json").write_text(json.dumps({
        "cutoff": str(snapshot.cutoff), "reference_ccy": snapshot.provenance["reference_ccy"],
        "covariance_units": "annual_asset_log_returns_squared", "residual_type": snapshot.residual_type,
        "factor_return_types": {name: kind.name for name, kind in snapshot.factor_return_types.items()},
        "factor_units": snapshot.provenance["factor_units"],
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


def public_proxy_prices(prices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a two-ETF proxy from supplied USD adjusted prices.

    None uses synthetic histories labelled SPY/TLT, not downloaded ETF evidence.
    Equity=SPY and Rates=TLT are simple long-only proxy definitions; they do not
    reproduce the proprietary MATF universe, hedges, carry or calibration.
    """
    from factormodels import Factor, FactorFamily, VolTargetPolicy, build_factor_panel
    from factormodels.spec import FactorModelSpec

    factor_prices, asset_prices = synthetic_prices()
    if prices is None:
        prices = factor_prices.rename(columns={"Equity": "SPY", "Rates": "TLT"})
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
    assets = asset_prices.reindex(levels.index).dropna()
    if len(assets) < 24:
        raise ValueError("The teaching asset panel requires at least 24 common dates in 2014–2024")
    levels = levels.loc[assets.index]
    return levels, assets


def run_local_test(case: int = 1) -> dict:
    """Run synthetic fit (1) or offline proxy construction/fit (2), returning objects."""
    if case not in (1, 2):
        raise ValueError("case must be 1 (synthetic) or 2 (synthetic ETF proxy)")
    result = inspect_fit(*(synthetic_prices() if case == 1 else public_proxy_prices()))
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
    parser.add_argument("--factor-prices", type=Path, help="Dated monthly USD total-return factor NAV CSV")
    parser.add_argument("--asset-prices", type=Path, help="Dated monthly USD total-return asset price CSV")
    parser.add_argument("--proxy-prices", type=Path, help="Already acquired USD adjusted SPY/TLT CSV")
    parser.add_argument("--acquire-proxy", action="store_true", help="Explicit optional Yahoo acquisition")
    parser.add_argument("--fit", action="store_true", help="Explicitly fit in memory")
    parser.add_argument("--output", type=Path, help="New output directory; requires --fit")
    args = parser.parse_args()
    if bool(args.factor_prices) != bool(args.asset_prices):
        parser.error("--factor-prices and --asset-prices must be supplied together")
    if sum(bool(x) for x in (args.factor_prices, args.proxy_prices, args.acquire_proxy)) > 1:
        parser.error("choose supplied factor/asset prices, --proxy-prices or --acquire-proxy")
    if args.output and not args.fit:
        parser.error("--output requires --fit")
    if args.factor_prices:
        factor_prices = pd.read_csv(resolve_resource_path(args.factor_prices), index_col=0, parse_dates=True)
        asset_prices = pd.read_csv(resolve_resource_path(args.asset_prices), index_col=0, parse_dates=True)
    elif args.acquire_proxy or args.proxy_prices:
        prices = (acquire_proxy_prices() if args.acquire_proxy else
                  pd.read_csv(resolve_resource_path(args.proxy_prices), index_col=0, parse_dates=True))
        factor_prices, asset_prices = public_proxy_prices(prices)
    else:
        factor_prices, asset_prices = synthetic_prices()
    print(f"Loaded {len(factor_prices)} factor observations")
    if args.fit:
        result = inspect_fit(factor_prices, asset_prices)
        print(result["snapshot"].asset_covariance)
        print(result["report_status"])
        if args.output:
            persist_example(result, args.output)


if __name__ == "__main__":
    main()
