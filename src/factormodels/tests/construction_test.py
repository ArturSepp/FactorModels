"""Independent Stage-4 accounting, causal risk and specification contracts.

All inputs are synthetic. No Rosaa installation, catalogue, providers or stored
private panels are needed. Frozen Rosaa parity is a separate explicit replay.
"""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from qis import ReturnTypes

from factormodels.construction import (
    build_factor_catalogue_panel,
    build_factor_panel,
    compute_factor_nav,
    compute_live_strategic_weights,
    compute_risk_factor_portfolio,
    compute_volatility_targeted_portfolio,
)
from factormodels.constraints import compute_beta_loading_signs
from factormodels.spec import (
    Factor,
    FactorFamily,
    FactorModelSpec,
    RiskFactors,
    VolTargetPolicy,
    assemble_factor_model,
    normalise_factor_labels,
)


def _weekly_prices(periods=60):
    """Return nonconstant correlated price paths with exactly weekly observations."""
    x = np.arange(periods, dtype=float)
    returns = np.column_stack((.012 * np.sin(x * .73) + .004 * np.cos(x * .29),
                               .008 * np.cos(x * .63) + .003 * np.sin(x * .73)))
    dates = pd.date_range("2020-01-01", periods=periods + 1, freq="W-WED")
    return pd.DataFrame(100 * np.exp(np.vstack((np.zeros(2), np.cumsum(returns, axis=0)))),
                        index=dates, columns=["A", "B"])


def _daily_prices(periods=180):
    """Return daily prices whose unequal paths distinguish held units from weights."""
    x = np.arange(periods, dtype=float)
    dates = pd.bdate_range("2025-01-02", periods=periods)
    return pd.DataFrame({"A": 100 * np.exp(.001 * x + .025 * np.sin(.2 * x)),
                         "B": 80 * np.exp(-.0003 * x + .018 * np.cos(.13 * x))}, index=dates)


# QIS owns production backtesting. This independent test oracle deliberately uses
# literal held units and cash to detect a same-day or constant-weight regression.
def _held_unit_nav(prices, weights, rebalance_dates):
    """Mark yesterday's units, then establish the next period's declared holdings."""
    values, units, cash, wealth = [], np.zeros(prices.shape[1]), 1., 1.
    for date, row in prices.iterrows():
        current = row.to_numpy()
        wealth = float(cash + units @ current)
        values.append(wealth)
        if date in rebalance_dates:
            units = wealth * weights / current
            cash = wealth * (1. - weights.sum())
    return pd.Series(values, index=prices.index)


@pytest.mark.parametrize("weights", [np.array([.6, .4]), np.array([1., -1.]), np.array([1.5, -1.])])
def test_quarterly_nav_matches_independent_lagged_units_and_cash(weights):
    """Long-only, net-zero and leveraged baskets retain quarterly held-unit accounting."""
    prices = _daily_prices(120)
    actual = compute_risk_factor_portfolio(prices, weights, rebalancing_freq="QE").get_portfolio_nav()
    expected = _held_unit_nav(prices, weights, {prices.index[0], pd.Timestamp("2025-03-31")})
    np.testing.assert_allclose(actual / actual.iloc[0], expected, rtol=1e-12, atol=1e-12)
    wrong_daily_weights = (1. + prices.pct_change().fillna(0.) @ weights).cumprod()
    assert np.max(np.abs(expected - wrong_daily_weights)) > 1e-5


def test_rebalance_day_return_uses_prior_units():
    """A rebalance-date price jump is earned by old holdings before weights reset."""
    dates = pd.to_datetime(["2025-03-28", "2025-03-31", "2025-04-01"])
    prices = pd.DataFrame({"A": [100., 200., 300.], "B": [100., 100., 50.]}, index=dates)
    actual = compute_risk_factor_portfolio(prices, np.array([.5, .5]), rebalancing_freq="QE")
    nav = actual.get_portfolio_nav()
    np.testing.assert_allclose(nav / nav.iloc[0], [1., 1.5, 1.5], rtol=1e-12)


def _target_weights(prices, **kwargs):
    """Construct weekly risk weights so initialization and updates remain observable."""
    return compute_volatility_targeted_portfolio(
        prices, np.array([1., -.4]), returns_freq="W-WED", rebalancing_freq="W-WED",
        portfolio_vol_target=.15, vol_span=156, **kwargs)


def test_thirteen_return_sample_seed_and_single_next_update():
    """Signed portfolio sample variance includes cross-covariance and uses ddof=1."""
    prices = _weekly_prices()
    weights = _target_weights(prices).reindex(prices.index).fillna(0.)
    basket = np.log(prices).diff().dropna() @ np.array([1., -.4])
    sample = basket.iloc[:13].to_numpy()
    initial_variance = sum((sample - sample.mean()) ** 2) / 12
    leverage = .15 / np.sqrt(52 * initial_variance)
    assert weights.iloc[:13].eq(0.).all().all()
    np.testing.assert_allclose(weights.iloc[13], leverage * np.array([1., -.4]), rtol=1e-12)
    decay = 1 - 2 / 157
    next_variance = decay * initial_variance + (1 - decay) * basket.iloc[13] ** 2
    np.testing.assert_allclose(weights.iloc[14],
                               .15 / np.sqrt(52 * next_variance) * np.array([1., -.4]), rtol=1e-12)


def test_future_prices_cannot_change_past_weights_or_nav():
    """Extreme future observations cannot contaminate the sample seed or EWMA history."""
    prices = _weekly_prices(90)
    cutoff = prices.index[37]
    changed = prices.copy()
    changed.loc[changed.index > cutoff] *= np.exp(np.linspace(1., 5., 53))[:, None]
    expected = _target_weights(prices.loc[:cutoff])
    pd.testing.assert_frame_equal(_target_weights(prices).loc[:cutoff], expected)
    pd.testing.assert_frame_equal(_target_weights(changed).loc[:cutoff], expected)
    factor = Factor("basket", FactorFamily("family", .15), ("A",), (1.,),
                    short_side_tickers=("B",), short_side_weights=(.4,), rebalancing_freq="W-WED")
    pd.testing.assert_series_equal(compute_factor_nav(factor, prices).loc[:cutoff],
                                    compute_factor_nav(factor, changed).loc[:cutoff])


def test_late_start_counts_observations_instead_of_empty_calendar_rows():
    """A late-starting factor gets a complete sample before any exposure."""
    prices = _weekly_prices()
    empty = pd.DataFrame(np.nan, index=pd.date_range(end=prices.index[0], periods=21,
                                                    freq="W-WED")[:-1], columns=prices.columns)
    actual = _target_weights(pd.concat([empty, prices]))
    expected = _target_weights(prices)
    assert actual.loc[:prices.index[12]].eq(0.).all().all()
    pd.testing.assert_frame_equal(actual.loc[expected.index], expected)


@pytest.mark.parametrize("warmup", [0, 1, -1, 3.5, True])
def test_invalid_sample_length_fails_in_spec_and_direct_constructor(warmup):
    """A covariance seed requires an integer sample length of at least two."""
    with pytest.raises(ValueError, match="vol_warmup_period"):
        Factor("basket", FactorFamily("family", .15), ("A",), (1.,), vol_warmup_period=warmup)
    with pytest.raises(ValueError, match="vol_warmup_period"):
        _target_weights(_weekly_prices(), vol_warmup_period=warmup)


def test_short_or_zero_risk_histories_hold_zero_exposure():
    """Unavailable or zero observed risk cannot create infinite leverage."""
    assert _target_weights(_weekly_prices(10)).eq(0.).all().all()
    prices = _weekly_prices()
    prices.iloc[:] = 100.
    assert _target_weights(prices).eq(0.).all().all()


def test_explicit_warmup_override_controls_first_traded_return():
    """The configured seed length affects actual holdings and is recorded in the recipe."""
    prices = _weekly_prices()
    factor = Factor("basket", FactorFamily("family", .15), ("A",), (1.,),
                    rebalancing_freq="W-WED", vol_warmup_period=8)
    actual = compute_factor_nav(factor, prices)
    delayed = compute_factor_nav(replace(factor, vol_warmup_period=13), prices)
    assert actual.loc[:prices.index[8]].eq(actual.iloc[0]).all()
    assert actual.loc[prices.index[9]] != pytest.approx(actual.iloc[0])
    assert delayed.loc[:prices.index[13]].eq(delayed.iloc[0]).all()
    assert "vol_warmup_period=8" in factor.to_text()
    assert "vol_init=warmup_sample_covariance" in factor.to_text()


def test_liveness_preserves_each_declared_side_and_empty_side_zero():
    """Missing legs redistribute within their side, preserving a net-zero basket."""
    prices = pd.DataFrame([[100., np.nan, 100., np.nan], [100., 100., 100., 100.],
                           [np.nan, np.nan, 100., 100.]], columns=list("ABCD"))
    actual = compute_live_strategic_weights(prices, np.array([.6, .4, -.3, -.7]))
    np.testing.assert_array_equal(actual, [[1., 0., -1., 0.], [.6, .4, -.3, -.7],
                                          [0., 0., -.3, -.7]])


def test_splice_preserves_newer_segment_exactly_and_scales_only_old_history():
    """An off-quarter inception preserves the new hedge's own holdings and NAV levels."""
    prices = _daily_prices(100).rename(columns={"A": "old", "B": "hedge"})
    prices["new"] = 70 * np.exp(.001 * np.arange(len(prices)))
    prices.loc[prices.index[:23], "new"] = np.nan
    family = FactorFamily("family", None)
    older = Factor("old factor", family, ("old",), (1.,), ("hedge",), (.4,),
                   rebalancing_freq="QE", require_all_constituents=True)
    newer = Factor("new factor", family, ("new",), (1.,), ("hedge",), (.4,),
                   rebalancing_freq="QE", require_all_constituents=True)
    original = compute_factor_nav(newer, prices)
    actual = compute_factor_nav(replace(newer, backfill_factor=older), prices)
    anchor = original.first_valid_index()
    pd.testing.assert_series_equal(actual.loc[anchor:], original.loc[anchor:], check_exact=True)
    history = compute_factor_nav(older, prices.loc[:anchor])
    expected_head = history.loc[history.index < anchor] / history.loc[anchor] * original.loc[anchor]
    np.testing.assert_allclose(actual.loc[expected_head.index], expected_head, rtol=1e-12)
    assert anchor == prices.index[23]


def test_joint_inception_rejects_later_constituent_gaps():
    """An inseparable hedged tracker waits for all legs and fails on a subsequent gap."""
    prices = _daily_prices(50)
    prices.loc[prices.index[:5], "A"] = np.nan
    factor = Factor("basket", FactorFamily("family", None), ("A", "B"), (.5, .5),
                    require_all_constituents=True)
    assert compute_factor_nav(factor, prices).first_valid_index() == prices.index[5]
    prices.loc[prices.index[20], "B"] = np.nan
    with pytest.raises(ValueError, match="after inception"):
        compute_factor_nav(factor, prices)


def test_composite_keeps_its_own_daily_anchor_under_outer_quarterly_override():
    """Outer override controls components; a daily constant mix stays a daily mix."""
    prices = _daily_prices(80)
    family = FactorFamily("family", None)
    a = Factor("a", family, ("A",), (1.,))
    b = Factor("b", family, ("B",), (1.,))
    composite = Factor("blend", family, ("a", "b"), (.5, .5),
                       rebalancing_freq="B", component_factors=(a, b))
    actual = compute_factor_nav(composite, prices, rebalancing_freq="QE")
    expected = (1. + .5 * prices.pct_change().fillna(0.).sum(axis=1)).cumprod()
    np.testing.assert_allclose(actual / actual.iloc[0], expected, rtol=1e-12, atol=1e-12)


class _Factors(RiskFactors):
    """Synthetic ordered factor set independent of any private catalogue."""
    BASKET = "basket"
    OBSERVED = "observed"


def test_production_parameters_are_explicit_and_reach_every_basket():
    """Uniform QE/span settings are caller choices; hidden defaults would change the NAV."""
    prices = _daily_prices(300)
    factor = Factor("basket", FactorFamily("family", .15), ("A", "B"), (.6, .4),
                    rebalancing_freq="ME", vol_span=26)
    observed = Factor("observed", FactorFamily("native", None), ("A",), (1.,),
                      vol_target_policy=VolTargetPolicy.NONE, is_observed_index=True,
                      return_type=ReturnTypes.DIFFERENCE)
    spec = assemble_factor_model("synthetic", _Factors, "synthetic", catalogue={
        "basket": factor, "observed": observed})
    panel = build_factor_panel(spec, prices, rebalancing_freq="QE", vol_span=156,
                               returns_freq="W-WED")
    expected = compute_factor_nav(factor, prices, rebalancing_freq="QE", vol_span=156,
                                   returns_freq="W-WED")
    pd.testing.assert_series_equal(panel.basket, expected)
    pd.testing.assert_series_equal(panel.observed, prices.A.rename("observed"))
    assert np.max(np.abs(panel.basket - compute_factor_nav(factor, prices))) > 1e-5
    assert list(panel.columns) == ["basket", "observed"]
    assert spec.return_types() == {"basket": ReturnTypes.LOG, "observed": ReturnTypes.DIFFERENCE}
    catalogue = build_factor_catalogue_panel(dict(spec.baskets), prices)
    assert list(catalogue.columns) == ["basket"]


class _Rules(RiskFactors):
    """Factor set covering sign, free and two shared-gate branches."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FREE = "free"
    GATED = "gated"
    GATED_SIGNED = "gated signed"


def test_shared_gate_opens_all_gated_factors_and_implies_other_signs():
    """One eligibility flag retains the existing shared-gate, sign and NaN semantics."""
    rules = [(1., False), (-1., False), (None, False), (None, True), (-1., True)]
    spec = FactorModelSpec("rules", _Rules, {
        member: Factor(member.value, FactorFamily(member.value, .1, sign, gated), ("A",), (1.,))
        for member, (sign, gated) in zip(_Rules, rules)}, "rules")
    index = ["plain", "signed", "gate", "both", "missing"]
    long_only = pd.Series([False, True, False, True, np.nan], index=index)
    gate = pd.Series([False, False, True, True, np.nan], index=index)
    actual = compute_beta_loading_signs(spec, long_only, gate)
    expected = pd.DataFrame([[np.nan, np.nan, np.nan, 0., 0.], [1., -1., np.nan, 0., 0.],
                             [1., -1., np.nan, 1., 1.], [1., -1., np.nan, 1., 1.],
                             [np.nan, np.nan, np.nan, 0., 0.]],
                            index=index, columns=spec.factor_names())
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    with pytest.raises(ValueError, match="does not cover"):
        compute_beta_loading_signs(spec, long_only, gate.drop("gate"))


def test_injected_aliases_change_only_supplied_labels():
    """Label normalization consumes caller policy and does not know a private catalogue."""
    values = pd.DataFrame([[1., 2.]], columns=["legacy", "untouched"])
    actual = normalise_factor_labels(values, aliases={"legacy": "canonical"})
    assert list(actual.columns) == ["canonical", "untouched"]
    assert list(values.columns) == ["legacy", "untouched"]
    np.testing.assert_array_equal(actual, values)


def test_construction_works_with_rosaa_imports_blocked():
    """A fresh Python process constructs a completed NAV with Rosaa imports impossible."""
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        class NoRosaa(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "ramen" or fullname.startswith("ramen."):
                    raise AssertionError("FM attempted a Rosaa import: " + fullname)
        sys.meta_path.insert(0, NoRosaa())
        import pandas as pd
        from factormodels import Factor, FactorFamily, compute_factor_nav
        prices = pd.DataFrame({"A": [100., 102., 99.]},
                              index=pd.bdate_range("2025-01-01", periods=3))
        result = compute_factor_nav(Factor("synthetic", FactorFamily("synthetic", None),
                                           ("A",), (1.,)), prices)
        assert list(result) == [100., 102., 99.]
        assert not any(name == "ramen" or name.startswith("ramen.") for name in sys.modules)
        print("NO_ROSAA_CONSTRUCTION_PASS")
    ''')
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(sys.executable).parent,
                            env=environment, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NO_ROSAA_CONSTRUCTION_PASS" in result.stdout
