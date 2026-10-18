"""Independent contracts for completed cluster mappings and pooled signal panels."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData, StabilityPoolingType, VarianceColumns

from tradablefactors.model import FactorModelSnapshot, RollingFactorModel
from tradablefactors.signals import extract_rolling_clusters, prepare_cluster_scores

TICKERS = [f"t{i}" for i in range(8)]
SIGNAL_DATES = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])


# --------------------------------------------------------------------------- #
# extract_rolling_clusters
# --------------------------------------------------------------------------- #
def make_rolling_covar_data(clusters_by_date: dict) -> SimpleNamespace:
    """The shape ``extract_rolling_clusters`` reads: ``.data[date].clusters``.

    ``RollingFactorCovarData`` comes out of a factorlasso estimation over real prices, which
    this suite deliberately does not run. Only the two attributes the function touches are
    stood up here, so the branches are exercised on stated assignments rather than on whatever
    an estimator happened to produce.
    """
    return SimpleNamespace(
        data={date: SimpleNamespace(clusters=clusters)
              for date, clusters in clusters_by_date.items()})


def test_extract_rolling_clusters_keys_the_assignments_by_estimation_date() -> None:
    """the flat per-date Series is passed through, keyed by the date it was estimated on"""
    assignment = pd.Series(['QE:1'] * 4 + ['QE:2'] * 4, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment,
                                 SIGNAL_DATES[1]: assignment}))
    assert list(rolling.keys()) == [SIGNAL_DATES[0], SIGNAL_DATES[1]]
    pd.testing.assert_series_equal(rolling[SIGNAL_DATES[0]], assignment)


def test_extract_rolling_clusters_skips_dates_with_no_assignment() -> None:
    """a date the estimator produced nothing for is absent, not present and empty

    An empty Series left in the dict would be scored as a partition of zero assets, which
    ``score_within_clusters`` reads as a degenerate cluster rather than as missing data.
    """
    assignment = pd.Series(['QE:1'] * 8, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: None,
                                 SIGNAL_DATES[1]: pd.Series(dtype=object),
                                 SIGNAL_DATES[2]: assignment}))
    assert list(rolling.keys()) == [SIGNAL_DATES[2]]


def test_extract_rolling_clusters_keeps_the_last_label_of_a_duplicated_ticker() -> None:
    """a ticker carrying both an ME and a QE label would otherwise form two clusters"""
    assignment = pd.Series(['ME:1', 'QE:3'], index=[TICKERS[0], TICKERS[0]])
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}))
    assert rolling[SIGNAL_DATES[0]].to_dict() == {TICKERS[0]: 'QE:3'}


def test_extract_rolling_clusters_filters_to_the_requested_universe() -> None:
    """the covariance universe is wider than the signal universe, so it is narrowed here"""
    assignment = pd.Series(['QE:1'] * 4 + ['QE:2'] * 4, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}), assets=TICKERS[:3])
    assert list(rolling[SIGNAL_DATES[0]].index) == TICKERS[:3]


def test_extract_rolling_clusters_drops_a_date_that_covers_none_of_the_universe() -> None:
    """filtering can empty a date, and an emptied date is dropped like an absent one"""
    assignment = pd.Series(['QE:1'] * 4, index=TICKERS[4:])
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}), assets=TICKERS[:3])
    assert rolling == {}


def test_extract_accepts_completed_tf_model_and_returns_independent_assignments() -> None:
    """The normal TF output supplies clusters without requiring consumer FL access."""
    date = SIGNAL_DATES[0]
    assignments = pd.Series(["ME:1"] * 5 + ["QE:2"] * 3, index=TICKERS)
    data = CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.04]], index=["factor"], columns=["factor"]),
        y_betas=pd.DataFrame(0.5, index=TICKERS, columns=["factor"]),
        y_variances=pd.DataFrame({VarianceColumns.RESIDUAL_VARS.value: 0.01}, index=TICKERS),
        estimation_date=date,
        clusters=assignments,
    )
    model = RollingFactorModel({date: FactorModelSnapshot(data, cutoff=date)})
    extracted = extract_rolling_clusters(model, assets=TICKERS[::-1])
    pd.testing.assert_series_equal(extracted[date], assignments.reindex(TICKERS[::-1]))
    extracted[date].iloc[0] = "changed"
    assert model.snapshots[date].data.clusters.loc[TICKERS[-1]] == "QE:2"


@pytest.mark.parametrize("pooling_type", list(StabilityPoolingType))
def test_unit_stability_preserves_independent_cluster_scores(pooling_type) -> None:
    """Unpooled and unit-weight scores match within/small-cluster arithmetic."""
    raw = pd.DataFrame([np.arange(8.)], index=SIGNAL_DATES[:1], columns=TICKERS)
    clusters = {SIGNAL_DATES[0]: pd.Series(["big"] * 5 + ["small"] * 3, index=TICKERS)}
    weights = pd.DataFrame(1., index=raw.index, columns=TICKERS)
    scores = prepare_cluster_scores(
        raw, clusters, pooling_type=pooling_type, stability_weights=weights,
    )
    expected = np.r_[(np.arange(5.) - 2.) / np.sqrt(2.5),
                     (np.arange(5., 8.) - 3.5) / np.sqrt(6.)]
    np.testing.assert_allclose(scores.iloc[0], expected, rtol=1e-14)


@pytest.mark.parametrize("pooling_type", ["cluster_variance", "asset_variance"])
def test_partial_stability_uses_requested_variance_pooling(pooling_type) -> None:
    """Nonuniform weights distinguish asset pooling from a cluster mean weight."""
    raw = pd.DataFrame([np.arange(8.)], index=SIGNAL_DATES[:1], columns=TICKERS)
    clusters = {SIGNAL_DATES[0]: pd.Series(["big"] * 5 + ["small"] * 3, index=TICKERS)}
    weights = pd.DataFrame([[0., .2, .4, .6, .8, 0., 0., 0.]],
                           index=raw.index, columns=TICKERS)
    scores = prepare_cluster_scores(
        raw, clusters, pooling_type=pooling_type, stability_weights=weights,
    )
    selected_weights = weights.iloc[0, :5].to_numpy()
    if pooling_type == "cluster_variance":
        selected_weights = selected_weights.mean()
    expected_large = (np.arange(5.) - 2.) / np.sqrt(
        selected_weights * 2.5 + (1. - selected_weights) * 6.,
    )
    expected = np.r_[expected_large, (np.arange(5., 8.) - 3.5) / np.sqrt(6.)]
    np.testing.assert_allclose(scores.iloc[0], expected, rtol=1e-14)


def test_unpooled_missing_partition_retains_clipped_population_score() -> None:
    """The empty-map fallback keeps the historical QIS scaling and clipping."""
    raw = pd.DataFrame([[-20., 1., 10.]], index=SIGNAL_DATES[:1], columns=TICKERS[:3])
    actual = prepare_cluster_scores(raw, {})
    clipped = np.array([-5., 1., 5.])
    np.testing.assert_allclose(actual.iloc[0], (clipped - clipped.mean()) / clipped.std())


def test_unpooled_mode_ignores_unused_stability_weights() -> None:
    """Disabled pooling never validates or applies optional stability evidence."""
    raw = pd.DataFrame([np.arange(8.)], index=SIGNAL_DATES[:1], columns=TICKERS)
    clusters = {SIGNAL_DATES[0]: pd.Series(["big"] * 5 + ["small"] * 3, index=TICKERS)}
    invalid_unused = pd.DataFrame(-1., index=raw.index, columns=TICKERS)
    pd.testing.assert_frame_equal(
        prepare_cluster_scores(raw, clusters),
        prepare_cluster_scores(raw, clusters, stability_weights=invalid_unused),
        check_exact=True,
    )


def test_future_assignments_and_stability_do_not_change_existing_scores() -> None:
    """As-of preparation uses neither a future partition nor future weights."""
    raw = pd.DataFrame([np.arange(8.)] * 3, index=SIGNAL_DATES, columns=TICKERS)
    clusters = {SIGNAL_DATES[1]: pd.Series(["big"] * 5 + ["small"] * 3, index=TICKERS)}
    weights = pd.DataFrame(.25, index=SIGNAL_DATES[1:2], columns=TICKERS)
    expected = prepare_cluster_scores(
        raw, clusters, pooling_type="asset_variance", stability_weights=weights,
    )
    future_date = pd.Timestamp("2024-04-30")
    future_clusters = {**clusters, future_date: pd.Series("future", index=TICKERS)}
    # Even a weight available by the March signal is too late for February's partition.
    future_weights = pd.concat([
        weights, pd.DataFrame(0., index=[SIGNAL_DATES[2], future_date], columns=TICKERS),
    ])
    actual = prepare_cluster_scores(
        raw, future_clusters, pooling_type="asset_variance", stability_weights=future_weights,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(actual.loc[SIGNAL_DATES[0]], np.zeros(len(TICKERS)))


def test_pooled_scores_reject_missing_or_out_of_range_weights() -> None:
    """Missing pooling evidence and invalid probabilities remain hard errors."""
    raw = pd.DataFrame([np.arange(8.)], index=SIGNAL_DATES[:1], columns=TICKERS)
    clusters = {SIGNAL_DATES[0]: pd.Series(["big"] * 5 + ["small"] * 3, index=TICKERS)}
    with pytest.raises(ValueError, match="stability_weights are required"):
        prepare_cluster_scores(raw, clusters, pooling_type="asset_variance")
    with pytest.raises(ValueError, match=r"weights must lie in \[0, 1\]"):
        prepare_cluster_scores(
            raw, clusters, pooling_type="cluster_variance",
            stability_weights=pd.DataFrame(2., index=raw.index, columns=TICKERS),
        )
