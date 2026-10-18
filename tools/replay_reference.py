"""Freeze and compare labelled numerical migration evidence without overwriting it.

Private paths resolve through TRADABLEFACTORS_STAGE0 or agents/stage0.json.
The manifest is public; numerical data and source exports remain local.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path):
    """Return a file's SHA-256 identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    """Load an existing fixture or private data package by explicit file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def labelled(value):
    """Convert labelled numerical values to stable, inspectable JSON content."""
    import numpy as np
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return {"kind": "frame", "index": [str(x) for x in value.index],
                "columns": [str(x) for x in value.columns], "values": labelled(value.to_numpy())}
    if isinstance(value, pd.Series):
        return {"kind": "series", "index": [str(x) for x in value.index],
                "name": str(value.name), "values": labelled(value.to_numpy())}
    if isinstance(value, np.ndarray):
        return labelled(value.tolist())
    if isinstance(value, dict):
        return {str(k): labelled(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [labelled(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def check(actual, expected, *, atol, rtol, path="root"):
    """Require exact structure/labels and tolerance-bounded numerical values."""
    import numpy as np
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys changed")
        for key in expected:
            check(actual[key], expected[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: length changed")
        for index, (a, e) in enumerate(zip(actual, expected)):
            check(a, e, atol=atol, rtol=rtol, path=f"{path}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                                   err_msg=path)
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def synthetic_inputs():
    """Reproduce OP's documented deterministic 72-month FCGL fixture."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(72026)
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    factors = rng.normal(.004, .035, size=(72, 3))
    b = np.array([[.9,.1,0],[.8,.2,0],[.7,.1,.1],[0,.8,.1],
                  [.1,.9,0],[0,.7,.2],[.1,0,.8],[0,.1,.9]])
    assets = factors @ b.T + rng.normal(0, .012, size=(72, 8))
    return (pd.DataFrame(100*np.exp(np.cumsum(factors, axis=0)), index=dates,
                         columns=["F1","F2","F3"]),
            {"ME": pd.DataFrame(assets, index=dates, columns=[f"A{i}" for i in range(8)])})


def current_content(data):
    """Expose economic components without coupling the reference to TF classes."""
    return {"betas": data.y_betas, "factor_covariance": data.x_covar,
            "variances": data.y_variances, "asset_covariance": data.get_y_covar()}


def estimate(implementation):
    """Compare current, rolling and historical-cutoff fits on identical inputs."""
    import numpy as np
    import qis
    from factorlasso import LassoModel, LassoModelType, VarianceColumns
    factors, returns = synthetic_inputs()
    if implementation == "reference":
        from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator
    else:
        from tradablefactors import FactorCovarEstimator

    def estimator():
        return FactorCovarEstimator(lasso_model=LassoModel(
            model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=3),
            rebalancing_freq="ME", factor_returns_freq="ME", factor_covar_span=24)

    cutoff = factors.index[-13]
    truncated_factors = factors.loc[:cutoff]
    truncated_returns = {k: v.loc[:cutoff] for k, v in returns.items()}
    period = qis.TimePeriod(factors.index[-8], factors.index[-1])
    if implementation == "reference":
        current = estimator().fit_current_factor_covars(factors, returns)
        past = estimator().fit_current_factor_covars(truncated_factors, truncated_returns,
                                                     estimation_date=cutoff)
        rolling = estimator().fit_rolling_factor_covars(factors, returns, period).data
    else:
        current = estimator().fit_current(factors, returns, cutoff=factors.index[-1]).data
        past = estimator().fit_current(factors, returns, cutoff=cutoff).data
        rolling = estimator().fit_rolling(factors, returns, time_period=period).data.data
    for data in [current, past, *rolling.values()]:
        b, f = data.y_betas.to_numpy(), data.x_covar.to_numpy()
        residual = data.y_variances[VarianceColumns.RESIDUAL_VARS.value].to_numpy()
        # Independent scalar summation rather than the production matrix assembly.
        reference = np.array([[sum(b[i,k]*f[k,l]*b[j,l]
                                   for k in range(b.shape[1]) for l in range(b.shape[1]))
                               for j in range(len(b))] for i in range(len(b))])
        reference += np.diag(residual)
        np.testing.assert_allclose(data.get_y_covar(), reference, rtol=1e-10, atol=1e-14)
    return {"current": current_content(current), "aligned_cutoff": current_content(past),
            "rolling": {str(k): current_content(v) for k,v in rolling.items()}}, current


def diagnostic_module(implementation):
    """Resolve the canonical owner for the requested implementation."""
    name = "optimalportfolios.matf.cma_diagnostics" if implementation == "reference" else "tradablefactors.cma.diagnostics"
    return importlib.import_module(name)


def diagnose(implementation, b=None, d=None, f=None, lam=None, m=None):
    """Capture GLS/CMA tables and validate independent QR/asset-space identities."""
    import numpy as np
    import pandas as pd
    module = diagnostic_module(implementation)
    if b is None:
        b = pd.DataFrame([[1,0],[0,1],[.5,.6],[1,-.2]], index=list("abcd"), columns=["equity","rates"])
        d = pd.Series([.01,.02,.03,.04], index=b.index)
        f = pd.DataFrame([[.04,.005],[.005,.01]], index=b.columns, columns=b.columns)
        lam = pd.Series([.05,.02], index=b.columns)
        m = b @ lam + pd.Series([.012,.015,-.011,.004], index=b.index)
    data = module.CmaDiagnosticInputs(betas=b, residual_variances=d, excess_cma=m,
                                     factor_covariance=f, factor_premia=lam)
    result = module.compute_cma_diagnostics(data)
    sigma = np.sqrt(d.to_numpy())
    q, _ = np.linalg.qr(b.to_numpy()/sigma[:,None])
    residual = sigma*((np.eye(len(b))-q@q.T)@(m.to_numpy()/sigma))
    covariance = b @ f @ b.T + np.diag(d)
    direct = float(m @ np.linalg.solve(covariance, m))
    np.testing.assert_allclose(result.assets.residual, residual, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(result.summary["total_sharpe2"], direct, atol=1e-11, rtol=1e-10)
    return {name: getattr(result, name) for name in
            ["summary","assets","factors","channels","scenarios","factor_contributions",
             "factor_sensitivity","factor_leave_one_out"]}


def paper_results(implementation, mapping):
    """Replay both preserved saved-fit cuts; no new fit or substitute input is used."""
    import pandas as pd
    results = {}
    for name, value in mapping.items():
        path = Path(value)
        b = pd.read_csv(path/"betas.csv", index_col=0)
        assets = pd.read_csv(path/"assets.csv", index_col=0)
        f = pd.read_csv(path/"factor_covar.csv", index_col=0)
        premia = pd.read_csv(path/"factor_premia.csv", index_col=0)
        lam = premia["base"].reindex(b.columns)
        d = assets.resid_vol.pow(2)
        m = assets.factor_excess_cma + assets.w_paper*assets.alpha
        results[name] = diagnose(implementation, b, d, f, lam, m)
    return results


def construction(implementation):
    """Replay the selected construction owner against independent held-unit accounting.

    Reference mode imports only the frozen Rosaa tree configured by ``main``.
    Candidate mode deliberately imports TF directly; allowing frozen Rosaa here
    would make a construction migration pass without executing its new owner.
    """
    import numpy as np
    import pandas as pd
    if implementation == "reference":
        from ramen.quant_model_mvp.rosaa.data.factors.construction import compute_risk_factor_portfolio
    elif implementation == "tradablefactors":
        from tradablefactors.construction import compute_risk_factor_portfolio
    else:
        raise NotImplementedError(
            "Consumer construction replay requires a real Rosaa pipeline binding; "
            "direct TF construction is not evidence of consumer cutover")
    dates = pd.bdate_range("2025-01-01", periods=95)
    prices = pd.DataFrame({"a":100*np.exp(np.arange(95)*.001),
                           "b":80*np.exp(np.arange(95)*-.0003)}, index=dates)
    result = compute_risk_factor_portfolio(prices, np.array([.6,.4]), rebalancing_freq="QE")
    # Explicit dollar holdings between QIS calendar decisions; units are lagged.
    import qis
    schedule = qis.generate_rebalancing_indicators(prices, freq="QE", include_start_date=True)
    wealth, units, values = 1., None, []
    for i, (_, row) in enumerate(prices.iterrows()):
        if units is not None:
            wealth = float(units @ row.to_numpy())
        values.append(wealth)
        if schedule.iloc[i]:
            units = wealth*np.array([.6,.4])/row.to_numpy()
    actual = result.nav.squeeze()
    np.testing.assert_allclose(actual/actual.iloc[0], values, rtol=1e-10, atol=1e-12)
    return {"nav": actual, "prices": prices}


def downstream(reference_root, data):
    """Capture actual OP optimisation and QIS supported stress report tables."""
    import numpy as np
    import pandas as pd
    from optimalportfolios.optimization.constraints import Constraints
    from optimalportfolios.optimization.general.quadratic import wrapper_quadratic_optimisation
    weights, outcome = wrapper_quadratic_optimisation(data.get_y_covar(), Constraints())
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-7)
    stress_fixture = load_file("_tf_reference_market", Path(reference_root)/
                              "QuantInvestStrats/src/qis/portfolio/stress/tests/conftest.py")
    from qis.portfolio.stress import (InstrumentLeg, InstrumentType, PortfolioHolding,
                                     ScenarioMode, ShockConvention, StressScenarios,
                                     run_portfolio_stress_test)
    portfolio = stress_fixture.market.__wrapped__()([
        PortfolioHolding("h", "Holding", 100., (InstrumentLeg(InstrumentType.DELTA_1,"actual",2.),))])
    shocks = StressScenarios(pd.DataFrame({"Equity":[-.2,0,.2]}, index=[-.2,0,.2]),
                             ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    result = run_portfolio_stress_test(portfolio, shocks, factor_grids={"equity":shocks})
    return {"optimisation_weights":weights, "stress_summaries":dict(result.summaries),
            "stress_grid_summaries":dict(result.grid_summaries), "stress_risk":result.risk}


def main():
    """Execute explicit capture/verify; absent evidence never counts as a pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["capture","verify"])
    parser.add_argument("--implementation", required=True, choices=["reference","tradablefactors","consumer"])
    parser.add_argument("--case", required=True, choices=["all","papers","rosaa"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.implementation == "consumer" and args.case in ("all", "rosaa"):
        parser.error("consumer construction replay has no Rosaa pipeline binding yet; "
                     "use tradablefactors for the library comparison and run Rosaa's integration gates separately")
    target = args.directory/f"{args.case}.json"
    if args.mode == "verify" and not target.exists() and args.case != "all":
        target = args.directory/"all.json"
    if args.mode == "capture" and target.exists():
        raise FileExistsError(f"Refusing to overwrite established baseline: {target}")
    if args.mode == "verify" and not target.is_file():
        raise FileNotFoundError(f"Missing established baseline: {target}")
    stage_path = Path(os.environ.get("TRADABLEFACTORS_STAGE0", Path(__file__).resolve().parents[1]/"agents/stage0.json"))
    stage = json.loads(stage_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    root = Path(stage["baseline_inputs"]["reference_root"])
    if digest(root/"sources.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest identity changed")
    if args.implementation == "reference":
        for package, version in stage["environments"]["reference_installed_versions"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Frozen reference environment changed: {package}")
        for item in json.loads((root/"sources.json").read_text(encoding="utf-8")):
            if digest(root/item["repo"]/item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen source/input changed: {item['repo']}/{item['path']}")
        installed_manifest = Path(stage["environments"]["installed_sources_manifest"])
        if digest(installed_manifest) != stage["environments"]["installed_sources_manifest_sha256"]:
            raise ValueError("Installed reference source manifest changed")
        for item in json.loads(installed_manifest.read_text(encoding="utf-8")):
            if digest(item["installed_path"]) != item["sha256"]:
                raise ValueError(f"Installed reference source changed: {item['installed_path']}")
    # Imports are explicit and frozen. Never use an arbitrary current checkout for reference.
    if args.implementation == "reference":
        sys.path.insert(0, str(root/"OptimalPortfolios/src"))
    if args.implementation == "reference":
        sys.path.insert(0, str(root/"Rosaa/src"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    identities = {"manifest_sha256":digest(args.manifest),
                  "sources_sha256":digest(root/"sources.json"),
                  "inputs":{key:{p.name:digest(p) for p in Path(value).iterdir() if p.is_file()}
                            for key,value in stage["baseline_inputs"]["paper_snapshots"].items()}}
    data = {}
    if args.case == "all":
        print("Replaying current/rolling estimates and aligned cutoff", flush=True)
        data["estimation"], fitted = estimate(args.implementation)
        print("Replaying GLS/CMA diagnostics and supplied-input consumers", flush=True)
        data["diagnostics"] = diagnose(args.implementation)
        data["downstream"] = downstream(root, fitted)
    if args.case in ("all","papers"):
        print("Replaying both preserved paper cuts", flush=True)
        data["papers"] = paper_results(args.implementation, stage["baseline_inputs"]["paper_snapshots"])
    if args.case in ("all","rosaa"):
        print(f"Replaying {args.implementation} construction and held-unit reference", flush=True)
        data["construction"] = construction(args.implementation)
    contents = {"identity":identities, "case":args.case, "data":labelled(data)}
    if args.case == "all":
        for artifact in manifest["expected_artifacts"]:
            value = data
            for key in artifact.split("."):
                value = value[key]
            if value is None:
                raise ValueError(f"Missing required artifact: {artifact}")
    if args.mode == "capture":
        if args.implementation != "reference":
            raise ValueError("Only the frozen reference can establish expected data")
        args.directory.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(contents, stream, indent=2, allow_nan=False)
        print(f"BASELINE CAPTURED {target}")
    else:
        expected = json.loads(target.read_text(encoding="utf-8"))
        if expected["case"] == "all" and args.case != "all":
            expected["case"] = args.case
            key = "papers" if args.case == "papers" else "construction"
            expected["data"] = {key: expected["data"][key]}
        check(contents, expected, **manifest["tolerances"])
        print(f"BASELINE PASS case={args.case} implementation={args.implementation} artifacts={len(data)}")


if __name__ == "__main__":
    main()
