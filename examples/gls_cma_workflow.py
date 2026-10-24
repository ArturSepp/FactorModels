"""Offline arithmetic GLS inference and a separate complete-forecast CMA audit.

Monthly simple excess-return NAVs, uniform weights, CLARABEL, no missing data.
The small example teaches the two stages; it does not substitute for MATF production
price preparation, expert priors, horizons or data. No network access. Persistence
requires an explicit --output directory; the log-return example is not converted.
"""

import argparse
import json
from pathlib import Path

from factormodels.utils.local_path import create_output_directory

import numpy as np
import pandas as pd
from scipy.stats import t

from factorlasso import LassoModel, LassoModelType
from factormodels.cma import CmaDiagnosticInputs, compute_cma_diagnostics
from factormodels.premia import infer_factor_premia


def arithmetic_example(*, residual_var_weight: float = 1.) -> dict:
    """Fit the arithmetic example and return GLS views, audit and their actual inputs.

    The selected diagonal residual policy scales annual residual variances by
    residual_var_weight. It does not estimate or diagonalize a correlated model.
    Cash is an annual 3% forecast; the historical NAVs already encode excess returns.
    """
    if not np.isfinite(residual_var_weight) or residual_var_weight <= 0:
        raise ValueError("residual_var_weight must be finite and strictly positive")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2006-09-30", periods=241, freq="ME")
    factors = ["Equity", "Rates", "Carry FX"]
    assets = [f"Index {i+1}" for i in range(8)]
    true_b = rng.uniform(-.2, 1., (8, 3))
    x = rng.normal(0., [.035, .015, .01], (240, 3))
    y = x @ true_b.T + rng.normal(0., .012, (240, 8)) + .0005
    factor_prices = pd.DataFrame(100*np.vstack([np.ones(3), np.cumprod(1+x, axis=0)]),
                                 index=dates, columns=factors)
    index_prices = pd.DataFrame(100*np.vstack([np.ones(8), np.cumprod(1+y, axis=0)]),
                                index=dates, columns=assets)
    x = factor_prices.pct_change(fill_method=None).iloc[1:]
    y = index_prices.pct_change(fill_method=None).iloc[1:]
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                       span=None, demean=True, solver="CLARABEL").fit(x=x, y=y)
    betas = model.coef_
    historical_residual = y - x @ betas.T
    # Arithmetic annualisation of monthly simple excess returns; no compounding bridge.
    alpha = 12*historical_residual.mean()
    residual_variances = residual_var_weight * (12*historical_residual.var(ddof=0))
    provider_cma = pd.Series([.08, .065, .06, .07, .05, .09, .055, .075], index=assets)
    views = {}
    for weight in (0., .5, 1.):
        views[weight] = infer_factor_premia(
            betas, residual_variances, provider_cma, cash_rate=.03,
            alpha=alpha, alpha_weight=weight)
    result = views[1.]

    # Independent weighted QR reference, not the production SVD implementation.
    design = betas.to_numpy()/np.sqrt(residual_variances.to_numpy())[:, None]
    target = (provider_cma-.03-alpha)/np.sqrt(residual_variances)
    q, r = np.linalg.qr(design, mode="reduced")
    reference = np.linalg.solve(r, q.T @ target)
    error = target - design @ reference
    inverse_r = np.linalg.solve(r, np.eye(3))
    covariance = float(error @ error)/5 * inverse_r @ inverse_r.T
    np.testing.assert_allclose(result.premia, reference, atol=1e-12)
    np.testing.assert_allclose(result.covariance, covariance, atol=1e-12)
    np.testing.assert_allclose(result.factors.upper,
                               reference+t.ppf(.975, 5)*np.sqrt(np.diag(covariance)))
    np.testing.assert_allclose(result.assets.fitted+result.assets.residual
                               +result.assets.cash+result.assets.credited_alpha, provider_cma)
    np.testing.assert_allclose(views[0.].premia-views[1.].premia,
                               result.mimicking_weights.T @ alpha)
    assert result.summary["rank"] == 3 and result.summary["residual_df"] == 5

    # The deliberately diagonal model uses annual arithmetic excess-return moments.
    factor_covariance = 12*x.cov(ddof=0)
    residual_covariance = pd.DataFrame(np.diag(residual_variances), index=assets, columns=assets)
    asset_covariance = betas @ factor_covariance @ betas.T + residual_covariance
    inputs = CmaDiagnosticInputs(
        betas=betas, residual_variances=residual_variances,
        factor_covariance=factor_covariance, production_covariance=asset_covariance,
        excess_cma=provider_cma-.03, factor_premia=result.premia,
        admitted_alpha=alpha,
        channels=pd.DataFrame({"admitted_alpha": alpha, "provider_residual": result.assets.residual}),
        provenance={"data_kind": "synthetic", "seed": 42, "reference_ccy": "USD",
                    "as_of": str(dates[-1].date()), "universe": "8 synthetic assets",
                    "model_name": "Arithmetic GLS/CMA teaching example",
                    "historical_return_basis": "monthly_simple_excess_decimal",
                    "annualisation_factor": 12, "covariance_ddof": 0,
                    "residual_type": "diagonal", "residual_var_weight": residual_var_weight,
                    "forecast_basis": "annual_arithmetic_total_decimal",
                    "cash_rate": .03, "calibrated_alpha_weight": 1.,
                    "audit_target": "provider_total_minus_cash; admitted alpha remains an additive channel"})
    audit = compute_cma_diagnostics(inputs)
    # Inference credits alpha; this audit projects the full cash-adjusted forecast.
    np.testing.assert_allclose(audit.factors.calibrated_premium, views[1.].premia, atol=1e-12)
    np.testing.assert_allclose(audit.factors.implied_premium, views[0.].premia, atol=1e-12)
    # Independent reference for the diagnostic's asset-space Sharpe accounting,
    # not portfolio attribution (which belongs to qis.RiskModel).
    chol = np.linalg.cholesky(asset_covariance)
    whitened_forecast = np.linalg.solve(chol, provider_cma-.03)
    np.testing.assert_allclose(audit.summary["total_sharpe2"],
                               whitened_forecast @ whitened_forecast, atol=1e-12)
    assert audit.summary["accounting_status"] == "pass"
    forecasts = pd.DataFrame({"provider_total": provider_cma, "cash": .03,
                              "credited_alpha": alpha, "gls_target": provider_cma-.03-alpha,
                              "audit_excess_target": provider_cma-.03})
    return {"premia": result, "views": views, "audit": audit, "inputs": inputs,
            "forecasts": forecasts, "factor_prices": factor_prices, "asset_prices": index_prices,
            "factor_returns": x, "asset_returns": y,
            "historical_residual": historical_residual, "asset_covariance": asset_covariance}


def persist_example(result: dict, directory: Path) -> Path:
    """Save completed arithmetic tables and a genuine PDF on explicit request.

    The exports retain exact inputs, forecast conventions and full precision.
    They are inspectable tables, not a fitted-object checkpoint.
    """
    from factormodels.reports.cma_diagnostics import render_cma_diagnostics

    directory = create_output_directory(directory)
    inputs, audit = result["inputs"], result["audit"]
    for name in ("forecasts", "factor_prices", "asset_prices", "factor_returns", "asset_returns",
                 "historical_residual", "asset_covariance"):
        result[name].to_csv(directory / f"{name}.csv")
    inputs.betas.to_csv(directory / "loadings.csv")
    inputs.factor_covariance.to_csv(directory / "factor_covariance.csv")
    inputs.residual_variances.to_csv(directory / "residual_variances.csv", header=["annual_variance"])
    inputs.channels.to_csv(directory / "construction_channels.csv")
    for name in audit.__dataclass_fields__:
        table = getattr(audit, name)
        if isinstance(table, pd.DataFrame):
            table.to_csv(directory / f"{name}.csv")
    for weight, view in result["views"].items():
        for name in ("assets", "factors", "covariance", "mimicking_weights"):
            getattr(view, name).to_csv(directory / f"gls_alpha_{weight:g}_{name}.csv")
    (directory / "audit_metadata.json").write_text(json.dumps({
        "summary": audit.summary, "provenance": audit.provenance,
        "gls_views": {f"alpha_{weight:g}": view.summary for weight, view in result["views"].items()},
    }, indent=2, default=str) + "\n", encoding="utf-8")
    render_cma_diagnostics(audit, directory / "cma_report.pdf", pages=2)
    return directory


def run_local_test(case: int = 1):
    """Run the offline arithmetic demonstration, preserving the GLS result interface."""
    if case != 1:
        raise ValueError('The synthetic GLS example supports only case 1.')
    completed = arithmetic_example()
    print(pd.DataFrame({f"alpha_{w:g}": value.premia for w, value in completed["views"].items()}))
    print(completed["premia"].factors[["estimate", "lower", "upper", "variance_inflation"]])
    return completed["premia"]


def main() -> None:
    """Compute offline; only an explicit output option writes the completed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for arithmetic GLS/CMA tables and PDF")
    args = parser.parse_args()
    result = arithmetic_example()
    print(result["audit"].factors[["calibrated_premium", "implied_premium"]])
    if args.output:
        persist_example(result, args.output)


if __name__ == "__main__":
    main()
