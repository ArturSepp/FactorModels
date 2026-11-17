"""Factor-only clustering anchors must never enter regression or portfolio outputs."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
import qis
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform

from factorlasso import (
    ClusterCorrelationTransform, ClusterSmootherType, LassoModel, LassoModelType,
    compute_rolling_smoothed_clusters,
    get_linkage_array,
)
from factormodels import FactorCovarEstimator


def _inputs():
    """Build a monthly panel with clean factor representatives and noisy assets."""
    rng = np.random.default_rng(912)
    dates = pd.date_range('2015-01-31', periods=60, freq='ME')
    x = pd.DataFrame(rng.normal(0, 0.02, (60, 3)), index=dates, columns=['F1', 'F2', 'F3'])
    y = pd.DataFrame({f'A{i}': x.iloc[:, i // 2] + rng.normal(0, 0.02, 60)
                      for i in range(6)}, index=dates)
    return np.exp(x.cumsum()), {'ME': y}


def _estimator(enabled=True, **model_kwargs):
    """Return a short, deterministic FCGL fit with automatically pooled signs."""
    settings = dict(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                    span=24, warmup_period=12, reg_lambda=1e-4, auto_sign_constraints=True)
    settings.update(model_kwargs)
    return FactorCovarEstimator(
        lasso_model=LassoModel(**settings),
        factor_returns_freq='ME', factor_covar_span=24, rebalancing_freq='ME',
        include_factors_in_clustering=enabled,
    )


@pytest.mark.parametrize('model_type', [
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
])
def test_anchor_fit_matches_independent_augmented_partition(model_type):
    """Match a separate clustering pass and fit only original response columns."""
    factors, returns = _inputs()
    estimator = _estimator()
    estimator = replace(estimator, lasso_model=estimator.lasso_model.copy(
        kwargs={'model_type': model_type}))
    y = returns['ME']
    x = qis.to_returns(factors, is_log_returns=True, is_first_zero=False,
                       drop_first=False, freq=None)
    date = y.index[-1]
    augmented = pd.concat([y, x.add_prefix('anchor::')], axis=1)
    partition = compute_rolling_smoothed_clusters(
        augmented, [date], estimator.lasso_model,
    )
    expected = estimator.lasso_model.copy().fit(
        x=x, y=y, external_clusters=partition.clusters[date].reindex(y.columns),
    )
    actual = estimator.fit_current(factors, returns, cutoff=date)
    pd.testing.assert_frame_equal(actual.loadings, expected.estimated_betas, check_exact=True)
    pd.testing.assert_frame_equal(actual.derived_signs, expected.derived_signs_, check_exact=True)
    assert actual.loadings.index.equals(y.columns)
    assert actual.loadings.columns.equals(x.columns)
    assert actual.residuals.columns.equals(y.columns)
    assert actual.asset_covariance.index.equals(y.columns)
    restricted = get_linkage_array(actual.linkages, 'ME')
    full_distances = squareform(cophenet(partition.linkages[date]))[:6, :6]
    np.testing.assert_array_equal(squareform(cophenet(restricted)), full_distances)
    fitted_labels = actual.clusters.to_numpy()
    tree_labels = fcluster(restricted, actual.cutoffs['ME'], criterion='distance')
    np.testing.assert_array_equal(
        fitted_labels[:, None] == fitted_labels,
        tree_labels[:, None] == tree_labels,
    )


def test_default_path_is_exactly_unchanged_and_flag_survives_copy():
    """False keeps the existing estimator, including its fitted-state contract."""
    factors, returns = _inputs()
    explicit = _estimator(False)
    default = FactorCovarEstimator(**{
        field.name: getattr(explicit, field.name) for field in fields(explicit)
        if field.name != 'include_factors_in_clustering'
    })
    expected = default.fit_current(factors, returns, cutoff=factors.index[-1])
    actual = explicit.fit_current(factors, returns, cutoff=factors.index[-1])
    pd.testing.assert_frame_equal(actual.loadings, expected.loadings, check_exact=True)
    pd.testing.assert_series_equal(actual.clusters, expected.clusters, check_exact=True)
    assert replace(_estimator()).include_factors_in_clustering is True


@pytest.mark.parametrize('smoother', [
    ClusterSmootherType.NONE, ClusterSmootherType.SIMILARITY_EWMA,
    ClusterSmootherType.PARTITION_BONUS, ClusterSmootherType.HOLD,
])
def test_rolling_anchors_are_causal_and_match_current(smoother):
    """Changing future prices and responses cannot change a past anchored fit."""
    factors, returns = _inputs()
    kwargs = {'cluster_smoother_type': smoother}
    if smoother == ClusterSmootherType.HOLD:
        kwargs['recluster_freq'] = 'QE'
    estimator = _estimator(**kwargs)
    start, end = factors.index[-5], factors.index[-2]
    period = qis.TimePeriod(start, end + pd.Timedelta(days=1))
    actual = estimator.fit_rolling(factors, returns, time_period=period)
    changed_factors = factors.copy()
    changed_factors.loc[changed_factors.index > end] *= 3
    changed_returns = {'ME': returns['ME'].copy()}
    changed_returns['ME'].loc[changed_returns['ME'].index > end] += 0.5
    repeated = _estimator(**kwargs).fit_rolling(
        changed_factors, changed_returns, time_period=period,
    )
    for date, expected in actual.snapshots.items():
        pd.testing.assert_frame_equal(expected.loadings, repeated.snapshots[date].loadings,
                                      check_exact=True)
        pd.testing.assert_series_equal(expected.clusters, repeated.snapshots[date].clusters)
    if smoother == ClusterSmootherType.NONE:
        current = estimator.fit_current(factors, returns, cutoff=end)
        pd.testing.assert_frame_equal(current.loadings, actual.snapshots[end].loadings, check_exact=True)


def test_linkage_restriction_preserves_cophenetic_distances_and_leaf_order():
    """Compare the induced tree against an independent cophenetic submatrix."""
    from factormodels.estimation._clustering import _restrict_linkage

    labels = pd.Index(list('abcdef'))
    tree = linkage(np.array([[0, 0], [0, 1], [2, 3], [2, 4], [5, 6], [5, 7]]), 'ward')
    for kept in [labels, labels[[5, 0, 3]], labels[:1], labels[:0]]:
        reduced = _restrict_linkage(tree, labels, kept)
        assert reduced.shape == (max(0, len(kept) - 1), 4)
        if len(kept) > 1:
            indices = labels.get_indexer(kept)
            expected = squareform(cophenet(tree))[np.ix_(indices, indices)]
            np.testing.assert_array_equal(squareform(cophenet(reduced)), expected)


def test_flag_rejects_unsupported_model_and_non_boolean_value():
    """Refuse flags whose requested behaviour cannot be honoured."""
    with pytest.raises(ValueError, match='HCGL or FCGL'):
        FactorCovarEstimator(lasso_model=LassoModel(), include_factors_in_clustering=True)
    with pytest.raises(TypeError, match='bool'):
        _estimator('yes')


@pytest.mark.parametrize('transform', list(ClusterCorrelationTransform))
def test_short_history_assets_do_not_pollute_reporting_leaves(transform):
    """The reporting tree and fitted clusters exclude responses below the warmup gate."""
    factors, returns = _inputs()
    returns['ME'].loc[returns['ME'].index[:-5], 'A0'] = np.nan
    with pytest.warns(UserWarning, match='warmup_period'):
        fitted = _estimator(cluster_correlation_transform=transform).fit_current(
            factors, returns, cutoff=factors.index[-1],
        )
    assert 'A0' not in fitted.clusters.index
    assert (fitted.loadings.loc['A0'] == 0).all()
    tree = get_linkage_array(fitted.linkages, 'ME')
    assert len(tree) + 1 == len(fitted.clusters)


def test_reference_input_guards_and_warmup_none():
    """Reject ambiguous columns while allowing an explicitly disabled warmup gate."""
    from factormodels.estimation._clustering import _restrict_linkage

    factors, returns = _inputs()
    actual = _estimator(warmup_period=None).fit_current(factors, returns, cutoff=factors.index[-1])
    assert len(actual.clusters) == 6
    repeated = pd.concat([factors, factors.iloc[:, :1]], axis=1)
    with pytest.raises(ValueError, match='unique'):
        _estimator().fit_current(repeated, returns, cutoff=factors.index[-1])
    collisions = {'ME': returns['ME'].rename(columns={'A0': '__factor_anchor__:0'})}
    with pytest.raises(ValueError, match='reserved'):
        _estimator().fit_current(factors, collisions, cutoff=factors.index[-1])
    with pytest.raises(ValueError, match='missing fitted'):
        _restrict_linkage(np.empty((0, 4)), pd.Index(['A']), pd.Index(['B']))


def test_singleton_reference_tree_renders():
    """A reference-pruned singleton has no merge rows and still renders its asset label."""
    import matplotlib.pyplot as plt

    labels, figure = qis.plot_clusters(
        clusters={'ME': pd.Series([1], index=['only asset'])},
        linkages={'ME': np.empty((0, 4))}, cutoffs={'ME': 0.5},
    )
    assert labels.index.tolist() == ['only asset']
    assert any(text.get_text() == 'only asset' for ax in figure.axes for text in ax.texts)
    plt.close(figure)


def test_monthly_and_quarterly_anchors_use_their_own_return_grids_and_spans():
    """Compare each cadence with a separate fit on independently aggregated factor returns."""
    factors, returns = _inputs()
    y = returns['ME']
    returns = {'ME': y.iloc[:, :3], 'QE': y.iloc[:, 3:].resample('QE').sum(min_count=1)}
    estimator = _estimator(span_freq_dict={'ME': 36, 'QE': 12})
    actual = estimator.fit_current(factors, returns, cutoff=factors.index[-1])
    for freq, responses in returns.items():
        prices = factors.reindex(responses.index, method='ffill').ffill()
        x = qis.to_returns(prices, is_log_returns=True, is_first_zero=False,
                           drop_first=False, freq=None)
        model = estimator.lasso_model.copy(kwargs={'span': {'ME': 36, 'QE': 12}[freq]})
        date = responses.index[-1]
        panel = pd.concat([responses, x.add_prefix('reference::')], axis=1)
        discovered = compute_rolling_smoothed_clusters(panel, [date], model)
        expected = model.fit(x=x, y=responses,
                             external_clusters=discovered.clusters[date].reindex(responses.columns))
        pd.testing.assert_frame_equal(actual.loadings.loc[responses.columns],
                                      expected.estimated_betas, check_exact=True)
        assert len(get_linkage_array(actual.linkages, freq)) == len(responses.columns) - 1


def test_explicit_cutoff_excludes_later_factor_prices():
    """The explicit valuation cutoff bounds the factor covariance despite later prices."""
    factors, returns = _inputs()
    future = factors.iloc[[-1]].copy() * 3.0
    future.index = pd.DatetimeIndex([factors.index[-1] + pd.offsets.MonthEnd()])
    expected = _estimator().fit_current(factors, returns, cutoff=factors.index[-1])
    actual = _estimator().fit_current(
        pd.concat([factors, future]), returns, cutoff=factors.index[-1])
    pd.testing.assert_frame_equal(actual.factor_covariance, expected.factor_covariance, check_exact=True)
    pd.testing.assert_frame_equal(actual.loadings, expected.loadings, check_exact=True)


@pytest.mark.parametrize('smoother', list(ClusterSmootherType))
def test_monthly_only_preserves_quarterly_fit_and_monthly_anchor_fit(smoother):
    """ME matches the all-cadence reference while QE exactly matches the original model."""
    factors, raw = _inputs()
    y = raw['ME']
    returns = {'ME': y.iloc[:, :3], 'QE': y.iloc[:, 3:].resample('QE').sum(min_count=1)}
    settings = dict(span_freq_dict={'ME': 36, 'QE': 12}, cluster_smoother_type=smoother)
    if smoother == ClusterSmootherType.HOLD:
        settings['recluster_freq'] = 'QE'
    baseline = _estimator(False, **settings)
    anchored = _estimator(True, **settings)
    selected = replace(_estimator(True, **settings), factor_clustering_freqs=['ME'])
    assert replace(selected).factor_clustering_freqs == ['ME']
    period = qis.TimePeriod(factors.index[-4], factors.index[-1] + pd.Timedelta(days=1))
    paths = [est.fit_rolling(factors, returns, time_period=period).snapshots
             for est in [baseline, anchored, selected]]
    current = [est.fit_current(factors, returns, cutoff=factors.index[-1])
               for est in [baseline, anchored, selected]]
    comparisons = [tuple(path[date] for path in paths) for date in paths[0]] + [current]
    for original, all_factors, monthly_only in comparisons:
        for freq, expected in [('ME', all_factors), ('QE', original)]:
            assets = returns[freq].columns
            pd.testing.assert_frame_equal(monthly_only.loadings.loc[assets],
                                          expected.loadings.loc[assets], check_exact=True)
            pd.testing.assert_frame_equal(monthly_only.derived_signs.loc[assets],
                                          expected.derived_signs.loc[assets], check_exact=True)
            pd.testing.assert_series_equal(monthly_only.clusters.reindex(assets),
                                           expected.clusters.reindex(assets), check_exact=True)
            np.testing.assert_array_equal(get_linkage_array(monthly_only.linkages, freq),
                                          get_linkage_array(expected.linkages, freq))
            pd.testing.assert_frame_equal(monthly_only.asset_covariance.loc[assets, assets],
                                          expected.asset_covariance.loc[assets, assets], check_exact=True)


@pytest.mark.parametrize('invalid', ['ME', [], [None], ['']])
def test_frequency_selector_rejects_ambiguous_values(invalid):
    """Cadence selection must be an explicit non-empty sequence of names."""
    with pytest.raises(ValueError, match='factor_clustering_freqs'):
        replace(_estimator(), factor_clustering_freqs=invalid)
