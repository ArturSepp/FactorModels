"""Synthetic pre-refactor cases; all analytics execute the public FM/FL/QIS APIs.

Only this candidate extraction layer changes as FM storage moves. Stable output
keys, input settings and frozen expected evidence must not be rewritten to make
a refactor pass. No providers, private inputs, optimizers or image comparisons.
"""
from __future__ import annotations

from dataclasses import make_dataclass

import numpy as np
import pandas as pd
import factorlasso as fl
import qis

import factormodels as fm

CONTRACT = {
    "schema": 1, "fixture": "fm-0.2.0-synthetic-v1",
    "seeds": {"construction": 2401, "estimation": 2092026, "native": 123,
              "loading_draws": 502, "alpha_uncertainty": 8},
    "solver": "CLARABEL", "reg_lambda": 1e-5, "span": 36,
    "span_by_frequency": {"ME": 36, "QE": 12}, "warmup": 6,
    "factor_frequency": "ME", "factor_span": 36,
    "current_cutoff": "2021-08-31", "rolling_start": "2021-06-30",
    "rolling_end": "2021-08-31", "residual_var_weight": .7,
    "nonorthogonal_corr_weight": .5, "alpha_span": 24,
    "alpha_span_by_frequency": {"ME": 24, "QE": 8},
    "policies": ["orthogonal", "empirical", "exposure_cluster", "residual_cluster"],
    "uncertainty_draws": 24, "loading_draws": 8,
}


def risk_values(risk):
    """Keep plain QIS interchange matrices and decomposition metadata only."""
    return {name: getattr(risk, name) for name in (
        "covar", "factor_loadings", "factor_covar", "residual_vars", "factor_return_types", "factor_groups")}


def snapshot_values(snapshot):
    """Extract the original economic schema from current FL-backed storage."""
    diagnostics = fm.prepare_snapshot_diagnostics(snapshot) if snapshot.fit_diagnostics is not None else None
    context = fm.prepare_portfolio_context(snapshot, diagnostics=diagnostics)
    info = snapshot.residual_correlation_metadata
    correlation = None
    if info is not None:
        fields = dict(correlation=snapshot.residual_correlation,
                      **{key: info[key] for key in ('residual_returns', 'asset_metadata', 'frequency',
                                                   'span', 'observation_date', 'estimation_date')})
        correlation = make_dataclass('CorrelationRecord', list(fields))(**fields)
    risk = context.pop("risk_model")
    result = {
        "cutoff": snapshot.cutoff, "residual_type": snapshot.residual_type,
        "residual_var_weight": snapshot.residual_var_weight,
        "residual_corr_weight": snapshot.residual_corr_weight,
        "factor_return_types": snapshot.factor_return_types,
        "factor_groups": snapshot.factor_groups, "provenance": snapshot.provenance,
        "loadings": snapshot.loadings, "factor_covariance": snapshot.factor_covariance,
        "selected_asset_covariance": snapshot.asset_covariance,
        "selected_residual_covariance": snapshot.residual_covariance,
        "raw_diagonal_reference": snapshot.raw_diagonal_reference_covariance,
        "fit_statistics": snapshot.fit_statistics, "residual_history": snapshot.residuals,
        "residual_metadata": snapshot.residual_metadata, "residual_correlation": correlation,
        "clusters": snapshot.clusters, "linkages": snapshot.linkages, "cutoffs": snapshot.cutoffs,
        "derived_signs": snapshot.derived_signs, "risk_export": risk_values(risk),
        "diagonal_reference_export": risk_values(snapshot.to_risk_model(diagonal_reference=True)),
        "clusters_by_frequency": snapshot.clusters_by_frequency,
        "linkages_by_frequency": snapshot.linkages_by_frequency,
        "cutoffs_by_frequency": snapshot.cutoffs_by_frequency,
        "portfolio_context": context,
    }
    if snapshot.residuals is not None:
        from factormodels.reports.covariance import _snapshot_table
        raw_table = snapshot.fit_diagnostics
        raw_table.insert(len(snapshot.loadings.columns)+1, 'stat_alpha', fm.estimate_historical_alpha(snapshot))
        result.update(alpha_default=fm.estimate_historical_alpha(snapshot),
                      alpha_scalar=fm.estimate_historical_alpha(snapshot, alpha_span=CONTRACT["alpha_span"]),
                      alpha_by_frequency=fm.estimate_historical_alpha(snapshot, alpha_span=CONTRACT["alpha_span_by_frequency"]),
                      fit_table=raw_table, report_table=_snapshot_table(snapshot, diagnostics))
    return result


def construction_case(inputs, outputs):
    """Freeze QIS-backed targeted and unscaled long/short factor construction."""
    rng = np.random.default_rng(CONTRACT["seeds"]["construction"])
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(.0002, .008, (320, 3)), axis=0)),
                          index=pd.date_range("2019-01-01", periods=320, freq="B"),
                          columns=["x", "y", "z"])
    family = fm.FactorFamily("synthetic", .12, long_only_sign=1.)
    factor = fm.Factor("spread", family, ("x", "y"), (.6, .4), ("z",), (.3,),
                       vol_span=26, vol_warmup_period=8)
    unscaled = fm.Factor("unscaled", family, ("x",), (1.,),
                         vol_target_policy=fm.VolTargetPolicy.NONE)
    inputs["construction"] = {"prices": prices, "factors": [factor, unscaled], "returns_freq": "W-WED"}
    outputs["construction"] = {f.name: fm.compute_factor_nav(f, prices, returns_freq="W-WED")
                                for f in (factor, unscaled)}


def estimation_cases(inputs, outputs):
    """Current and rolling fits on monthly/quarterly histories and all policies."""
    rng = np.random.default_rng(CONTRACT["seeds"]["estimation"])
    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    x = pd.DataFrame(rng.normal(.002, .035, (108, 2)), index=dates, columns=["f", "g"])
    common = rng.normal(0., .02, len(dates))
    monthly = pd.DataFrame({"a": .7*x.f + .1*x.g + common,
                            "b": .2*x.f - .3*x.g + .8*common + rng.normal(0., .006, len(dates))})
    quarterly = pd.DataFrame({"c": (.4*x.f + rng.normal(0., .02, len(dates))).resample("QE").sum()})
    quarterly["d"] = .5*quarterly.c + .003*np.sin(np.arange(len(quarterly)))
    prices, returns = np.exp(x.cumsum()), {"ME": monthly, "QE": quarterly}
    lasso = fl.LassoModel(model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                         span=36, span_freq_dict={"ME": 36, "QE": 12},
                         warmup_period=6, reg_lambda=1e-5, solver="CLARABEL", auto_sign_constraints=True)
    inputs["estimation"] = {"prices": prices, "returns": returns, "lasso": lasso.get_params()}
    for policy in CONTRACT["policies"]:
        estimator = fm.FactorCovarEstimator(
            lasso_model=lasso, factor_returns_freq="ME", factor_covar_span=36,
            rebalancing_freq="ME", residual_type=policy, residual_var_weight=.7,
            residual_corr_weight=1. if policy == "orthogonal" else .5,
            include_factors_in_clustering=True,
        )
        current = estimator.fit_current(prices, returns, cutoff=pd.Timestamp(CONTRACT["current_cutoff"]))
        outputs[f"current_{policy}"] = snapshot_values(current)
        rolling = estimator.fit_rolling(prices, returns, time_period=qis.TimePeriod(
            CONTRACT["rolling_start"], CONTRACT["rolling_end"]))
        clusters = fm.extract_rolling_clusters(rolling)
        signal = monthly.join(quarterly).reindex(list(rolling.snapshots)).fillna(0.)
        outputs[f"rolling_{policy}"] = {
            "snapshots": {date: snapshot_values(s) for date, s in rolling.snapshots.items()},
            "clusters": clusters, "scores": fm.prepare_cluster_scores(signal, clusters, min_cluster_size=1),
            "risk_export": risk_values(rolling.to_risk_model()),
        }


def native_case(inputs, outputs):
    """Difference-return coordinates with estimated and supplied factor covariance."""
    rng = np.random.default_rng(CONTRACT["seeds"]["native"])
    dates = pd.date_range("2015-01-31", periods=80, freq="ME")
    levels = pd.DataFrame({"native": rng.uniform(12., 45., len(dates))}, index=dates)
    returns = pd.DataFrame(levels.diff().to_numpy() * np.array([-.002, .001, -.004]),
                           index=dates, columns=["a", "b", "c"]).dropna()
    lasso = fl.LassoModel(span=24, warmup_period=12, reg_lambda=0., demean=False, solver="CLARABEL")
    estimator = fm.FactorCovarEstimator(lasso, factor_returns_freq="ME", factor_covar_span=24,
                                       factor_return_types={"native": qis.ReturnTypes.DIFFERENCE})
    supplied = fm.DatedFactorCovariance(pd.DataFrame([[400.]], index=["native"], columns=["native"]),
                                        dates[-2], dates[-1], {"native": qis.ReturnTypes.DIFFERENCE})
    inputs["native"] = {"levels": levels, "returns": returns, "lasso": lasso.get_params(), "supplied": supplied}
    outputs["native_estimated"] = snapshot_values(estimator.fit_current(levels, {"ME": returns}, cutoff=dates[-1]))
    outputs["native_supplied"] = snapshot_values(estimator.fit_current(
        levels, {"ME": returns}, cutoff=dates[-1], factor_covariance=supplied))


def diagnostic_cases(inputs, outputs):
    """Labelled annual GLS, CMA, opportunity and conditional uncertainty outputs."""
    ids, factors = pd.Index(list("abcdef")), pd.Index(["equity", "rates"])
    b = pd.DataFrame([[1., 0.], [0., 1.], [.5, .6], [1., -.2], [.2, .8], [.8, .1]], index=ids, columns=factors)
    d = pd.Series([.01, .02, .03, .04, .025, .015], index=ids, name="residual_var")
    f = pd.DataFrame([[.04, .005], [.005, .01]], index=factors, columns=factors)
    premia = pd.Series([.05, .02], index=factors)
    alpha = pd.Series([.01, .015, -.01, .003, -.005, .02], index=ids)
    channels = pd.DataFrame({"regional": [.002, 0., -.001, .001, .003, -.002], "admitted_alpha": alpha}, index=ids)
    excess = b @ premia + channels.sum(axis=1)
    covariance = pd.DataFrame(np.diag(np.full(len(ids), .000025)), index=ids, columns=ids)
    cma = fm.CmaDiagnosticInputs(b, d, excess, f, premia, channels, admitted_alpha=alpha,
                                 scenarios=pd.DataFrame({"base": excess, "up": excess + .005}),
                                 provenance={"currency": "USD", "fixture": "synthetic", "units": "annual_decimal"})
    inputs["diagnostics"] = {"cma": cma, "alpha_covariance": covariance, "cash_rate": .03}
    outputs["gls"] = {
        "geometry": fm.compute_gls_geometry(b, d),
        "premia": fm.infer_factor_premia(b, d, excess + .03, cash_rate=.03, alpha=alpha, alpha_weight=.5),
        "resolution": fm.compute_gls_resolution(b, d),
        "uncertainty": fm.compute_gls_uncertainty(b, d, premia, covariance, distribution="normal"),
        "residual_opportunity": fm.compute_residual_opportunity(b, d, alpha, controls=np.ones((len(ids), 1))),
    }
    outputs["cma"] = {"audit": fm.compute_cma_diagnostics(cma, policy=fm.AdmissionPolicy()),
                      "identification": fm.compute_cma_identification(cma, factor_pair=tuple(factors)),
                      "conditional_intervals": fm.compute_conditional_gls_intervals(cma),
                      "zero_admission": fm.compute_cma_diagnostics(fm.without_admitted_alpha(cma))}
    rng = np.random.default_rng(CONTRACT["seeds"]["loading_draws"])
    b_draws = b.to_numpy() + rng.normal(0., .02, (8, *b.shape))
    d_draws = np.tile(d.to_numpy(), (8, 1))
    inputs["loading_draws"] = {"betas": b_draws, "variances": d_draws}
    outputs["loading_stability"] = fm.compute_gls_loading_stability(
        b, d, excess + .03, B_draws=b_draws, Omega_draws=d_draws,
        cash_rate=.03, alpha=alpha, alpha_weight=.5, baseline_provenance="synthetic fixed model")
    cutoff = pd.Timestamp("2021-08-31")
    minimal = fm.snapshot_from_moments(b, f, d, date=cutoff)
    outputs["minimal_snapshot"] = snapshot_values(minimal)
    correlation = pd.DataFrame(.4 + .6*np.eye(len(ids)), index=ids, columns=ids)
    prepared = fl.ResidualCorrelationData(correlation=correlation,
        residual_returns=pd.DataFrame([alpha.to_numpy()], index=[cutoff], columns=ids),
        asset_metadata=pd.DataFrame(index=ids), frequency="ME", span=24,
        observation_date=cutoff, estimation_date=cutoff)
    inputs["opportunity_correlation"] = prepared
    for name, corr in (("diagonal", None), ("correlated", prepared)):
        snapshot = fm.snapshot_from_moments(b, f, d, date=cutoff, residual_correlation=corr,
                                            residual_type='empirical' if corr is not None else 'orthogonal')
        model = fm.FactorOpportunityModel.from_snapshot(snapshot, alpha)
        outputs[f"opportunity_{name}"] = {
            "prepared_inputs": model.optimisation_inputs(),
            "projected_alpha_covariance": model.prepare_alpha_covariance(covariance, orthogonal=True),
            "dispersion": fm.alpha_dispersion(model), "access": fm.factor_access_metrics(model, premia),
            "concentration": fm.dispersion_concentration(model), "removal": fm.removal_diagnostics(model),
            "subset": fm.subset_dispersion(model, list("abcd")),
            "uncertainty": fm.alpha_uncertainty_metrics(model, covariance, draws=24, seed=8),
        }


def collect() -> tuple[dict, dict]:
    """Return inputs and stable economic outputs without writing any artifact."""
    inputs, outputs = {}, {}
    construction_case(inputs, outputs)
    estimation_cases(inputs, outputs)
    native_case(inputs, outputs)
    diagnostic_cases(inputs, outputs)
    return inputs, outputs
