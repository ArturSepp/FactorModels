"""Independent exact FL materialization and pure completed-result contracts."""
from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
import factorlasso as fl
import factormodels as fm


def raw_fixture(*, history=True, full=True):
    """Declared mixed-cadence residual units and distinct correlation dates."""
    assets, factors = pd.Index(['z', 'a', 'm']), pd.Index(['rates', 'equity'])
    dates = pd.date_range('2024-01-31', periods=8, freq='ME')
    residuals = pd.DataFrame([[np.nan, np.nan, np.nan], [.1, np.nan, -.2],
        [.2, .3, -.1], [np.nan, np.nan, -.05], [.15, np.nan, .02],
        [.13, -.1, .03], [.11, np.nan, .05], [.09, np.nan, .07]], index=dates, columns=assets)
    metadata = pd.DataFrame({'frequency': ['ME', 'QE', 'ME'], 'beta_span': [24, 8, 24],
        'annualisation_factor': [12., 4., 12.], 'residual_scale': [12., 4., 12.]}, index=assets)
    stats = pd.DataFrame({'residual_var': [.01, .04, .02]}, index=assets)
    if full:
        stats['r2'], stats['insample_alpha'] = [.4, .5, .6], [.01, -.02, .03]
    corr = fl.ResidualCorrelationData(pd.DataFrame(.3+.7*np.eye(3), index=assets, columns=assets),
        residuals.iloc[[2, 5]].fillna(0.), metadata.copy(), 'QE', 8., dates[5], dates[6])
    return fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[.03, -.005], [-.005, .08]], index=factors, columns=factors),
        y_betas=pd.DataFrame([[.2, .8], [-.3, .5], [.4, -.1]], index=assets, columns=factors),
        y_variances=stats, estimation_date=dates[-1], residuals=residuals if history else None,
        residual_metadata=metadata if history else None, residual_correlation=corr,
        clusters=pd.Series(['ME:1', 'QE:1', 'ME:1'], index=assets),
        derived_signs=pd.DataFrame([[1., 1.], [-1., 1.], [1., -1.]], index=assets, columns=factors),
    )


def completed(raw, policy='empirical', weight=.35):
    """Select the risk policy explicitly at the sole FL materialization boundary."""
    return fm.snapshots_from_factorlasso(raw, residual_type=policy, residual_var_weight=weight,
        residual_corr_weight=1. if policy == 'orthogonal' else .5).snapshots[raw.estimation_date]


@pytest.mark.parametrize('policy', ['orthogonal', 'empirical', 'exposure_cluster', 'residual_cluster'])
def test_exact_materialization_keeps_raw_fit_and_three_risk_views(policy):
    """FL itself is the reference; raw D must not be replaced with weighted diag(R)."""
    raw = raw_fixture()
    snapshot = completed(raw, policy)
    options = dict(residual_type=policy, residual_var_weight=.35,
                   residual_corr_weight=1. if policy == 'orthogonal' else .5)
    for actual, expected in [(snapshot.loadings, raw.y_betas),
        (snapshot.factor_covariance, raw.x_covar), (snapshot.fit_statistics, raw.y_variances),
        (snapshot.residuals, raw.residuals), (snapshot.residual_metadata, raw.residual_metadata),
        (snapshot.residual_covariance, raw.get_residual_covar(**options)),
        (snapshot.asset_covariance, raw.get_y_covar(**options)),
        (snapshot.raw_diagonal_reference_covariance, raw.get_y_covar(residual_type='orthogonal'))]:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert 'stat_alpha' not in snapshot.fit_diagnostics
    assert snapshot.residual_correlation_metadata['observation_date'] < snapshot.residual_correlation_metadata['estimation_date'] < snapshot.cutoff
    np.testing.assert_array_equal(snapshot.fit_statistics.residual_var, raw.y_variances.residual_var)
    assert not np.array_equal(snapshot.fit_statistics.residual_var, np.diag(snapshot.residual_covariance))


@pytest.mark.parametrize('options', [{}, {'alpha_span': 24}, {'alpha_span': {'ME': 24, 'QE': 8}},
    {'alpha_span': {'ME': 24, 'QE': 8}, 'asset_frequencies': pd.Series({'a': 'QE'}), 'default_freq': 'ME'},
    {'alpha_span': {'ME': 24}, 'asset_frequencies': 'ME'}])
def test_alpha_delegation_is_exact_for_spans_native_units_and_missing_events(options):
    """No annual multiplier or alternative EWMA enters the adapter."""
    raw = raw_fixture()
    snapshot = completed(raw)
    pd.testing.assert_series_equal(fm.estimate_historical_alpha(snapshot, **options),
                                    raw.estimate_alpha(**options), check_exact=True)
    expected = raw.get_snapshot(**options)
    expected['total_vol'] = np.sqrt(np.diag(snapshot.asset_covariance))
    expected['resid_vol'] = np.sqrt(np.diag(snapshot.residual_covariance))
    prepared = fm.prepare_snapshot_diagnostics(snapshot, **options)
    pd.testing.assert_frame_equal(prepared.table, expected, check_exact=True)
    assert prepared.alpha_source == 'residual_history' and prepared.alpha_available


def test_missing_history_fallback_and_minimal_moments_are_explicit():
    """Preserve FL fallback and avoid manufacturing fit evidence for supplied moments."""
    raw = raw_fixture(history=False)
    snapshot = completed(raw)
    pd.testing.assert_series_equal(fm.estimate_historical_alpha(snapshot), raw.get_snapshot().stat_alpha)
    assert fm.prepare_snapshot_diagnostics(snapshot).alpha_source == 'insample_alpha'
    minimal = fm.snapshot_from_moments(raw.y_betas, raw.x_covar, raw.y_variances.residual_var,
                                       date=raw.estimation_date)
    assert minimal.residuals is minimal.residual_metadata is minimal.fit_diagnostics is None
    assert list(minimal.fit_statistics) == ['residual_var']
    prepared = fm.prepare_snapshot_diagnostics(minimal)
    assert prepared.alpha_source == 'unavailable' and not prepared.alpha_available
    assert prepared.table.stat_alpha.isna().all()
    assert 'r2' not in prepared.table and 'insample_alpha' not in prepared.table
    with pytest.raises(ValueError, match='unavailable'):
        fm.estimate_historical_alpha(minimal)
    assert fm.prepare_portfolio_context(minimal)['risk_model'] is not None


def test_all_completed_public_fields_are_defensive_copies():
    """Mutation of input/output matrices, maps, nested metadata and tables is isolated."""
    raw = raw_fixture()
    snapshot = completed(raw).with_provenance({'source': {'currency': 'USD'}})
    for name in ('loadings', 'factor_covariance', 'asset_covariance', 'residual_covariance',
                 'raw_diagonal_reference_covariance', 'fit_statistics', 'fit_diagnostics',
                 'residuals', 'residual_metadata', 'residual_correlation', 'derived_signs'):
        before = getattr(snapshot, name).copy(deep=True)
        displayed = getattr(snapshot, name)
        displayed.iloc[0, 0] = displayed.iloc[-1, 0]
        displayed.index.name = 'changed'
        pd.testing.assert_frame_equal(getattr(snapshot, name), before, check_exact=True)
    raw.y_betas.iloc[0, 0] = 999.
    assert snapshot.loadings.iloc[0, 0] != 999.
    snapshot.provenance['source']['currency'] = 'EUR'
    assert snapshot.provenance['source']['currency'] == 'USD'
    info = snapshot.residual_correlation_metadata
    info['asset_metadata'].iloc[0, 1] = 999.
    assert snapshot.residual_correlation_metadata['asset_metadata'].iloc[0, 1] != 999.
    clusters = snapshot.clusters_by_frequency
    clusters['ME'].iloc[0] = 'mutated'
    assert snapshot.clusters_by_frequency['ME'].iloc[0] != 'mutated'
    rolling = fm.RollingFactorModel({snapshot.cutoff: snapshot})
    rolling.snapshots.clear()
    assert len(rolling.snapshots) == 1
    prepared = fm.prepare_snapshot_diagnostics(snapshot, alpha_span={'ME': 24, 'QE': 8})
    table = prepared.table
    table.iloc[0, 0] = 999.
    prepared.alpha_settings['alpha_span']['ME'] = 999
    assert prepared.table.iloc[0, 0] != 999.
    assert prepared.alpha_settings['alpha_span']['ME'] == 24


def assert_no_fl(value):
    """Recursively examine stored fields, including private payloads and pandas attrs."""
    assert not type(value).__module__.startswith('factorlasso')
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_fl(key)
            assert_no_fl(item)
    elif is_dataclass(value):
        for field in fields(value):
            assert_no_fl(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_fl(item)
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        assert_no_fl(value.attrs)
        if isinstance(value, pd.DataFrame):
            for item in value.loc[:, value.dtypes == object].to_numpy().flat:
                assert_no_fl(item)


def test_completed_inspection_and_reports_call_neither_fl_nor_estimation(monkeypatch):
    """All preparation precedes disabling the statistical producer."""
    from factormodels.reports.covariance import run_rolling_covar_report
    raw = raw_fixture()
    snapshot = completed(raw)
    prepared = fm.prepare_snapshot_diagnostics(snapshot)
    rolling = fm.RollingFactorModel({snapshot.cutoff: snapshot})
    for item in (snapshot, rolling, prepared):
        assert_no_fl(item)
        assert not hasattr(item, 'data')
    def forbidden(*args, **kwargs):
        raise AssertionError('completed inspection called FL or fitting')
    for name in ('get_y_covar', 'get_residual_covar', 'get_snapshot', 'get_model_vols', 'estimate_alpha'):
        monkeypatch.setattr(fl.CurrentFactorCovarData, name, forbidden)
    monkeypatch.setattr(fm.FactorCovarEstimator, 'fit_current', forbidden)
    monkeypatch.setattr(fm.FactorCovarEstimator, 'fit_rolling', forbidden)
    assert snapshot.asset_covariance.shape == (3, 3)
    snapshot.to_risk_model()
    rolling.to_risk_model(diagonal_reference=True)
    fm.extract_rolling_clusters(rolling)
    context = fm.prepare_portfolio_context(snapshot, diagnostics=prepared)
    figs, tables = run_rolling_covar_report(rolling, diagnostics={snapshot.cutoff: prepared}, is_plot=False)
    assert figs == [] and len(tables) == 1
    pd.testing.assert_frame_equal(context['asset_diagnostics'], prepared.table)
    fm.FactorOpportunityModel.from_snapshot(snapshot, prepared.table.stat_alpha).optimisation_inputs()


def test_alignment_availability_and_raw_adapter_errors_are_not_deferred():
    """Reject incompatible dates/labels before returning a completed object."""
    raw = raw_fixture()
    snapshot = completed(raw)
    with pytest.raises(ValueError, match='same order|align'):
        completed(replace(raw, x_covar=raw.x_covar.iloc[::-1, ::-1]))
    with pytest.raises(ValueError, match='cutoff'):
        fm.snapshots_from_factorlasso({raw.estimation_date-pd.Timedelta(days=1): raw}, residual_type='orthogonal')
    future_corr = replace(raw.residual_correlation, estimation_date=raw.estimation_date+pd.Timedelta(days=1))
    with pytest.raises(ValueError, match='available|cutoff'):
        completed(replace(raw, residual_correlation=future_corr), policy='orthogonal')
    with pytest.raises(ValueError, match='history extends'):
        completed(replace(raw, residuals=raw.residuals.rename(index={raw.estimation_date: raw.estimation_date+pd.Timedelta(days=1)})))
    with pytest.raises(TypeError, match='residual_var_weight'):
        fm.build_risk_model(snapshot, residual_var_weight=.5)
    with pytest.raises(ValueError, match='snapshots_from_factorlasso'):
        fm.build_risk_model({raw.estimation_date: raw})
    prepared = fm.prepare_snapshot_diagnostics(snapshot)
    wrong = fm.PreparedSnapshotDiagnostics(prepared.table.iloc[::-1], cutoff=snapshot.cutoff,
        alpha_settings={}, alpha_source='supplied', alpha_available=True)
    with pytest.raises(ValueError, match='cutoff and asset order'):
        fm.prepare_portfolio_context(snapshot, diagnostics=wrong)
    with pytest.raises(TypeError, match='FactorLasso'):
        snapshot.with_provenance({'nested': [raw]})


def test_opportunity_preserves_raw_d_and_selected_weighted_correlated_risk():
    """GLS gets raw D, risk gets selected R/Sigma, and reference risk stays unit weight."""
    raw = raw_fixture()
    snapshot = completed(raw)
    model = fm.FactorOpportunityModel.from_snapshot(snapshot, fm.estimate_historical_alpha(snapshot))
    pd.testing.assert_series_equal(model.residual_variances, raw.y_variances.residual_var)
    pd.testing.assert_frame_equal(model.factor_risk_model.covar[snapshot.cutoff],
                                   raw.get_y_covar(residual_type='orthogonal'), check_exact=True)
    inputs = model.optimisation_inputs()
    pd.testing.assert_frame_equal(inputs['residual_covariance'], snapshot.residual_covariance, check_exact=True)
    pd.testing.assert_frame_equal(inputs['covariance'], snapshot.asset_covariance, check_exact=True)
    with pytest.raises(ValueError, match='views disagree'):
        replace(model, asset_covariance=model.raw_diagonal_reference_covariance)


def test_identical_owned_covariance_views_share_storage_but_reads_do_not():
    """Avoid a redundant rolling N-by-N buffer without exposing aliases."""
    snapshot = completed(raw_fixture(), policy='orthogonal', weight=1.)
    assert snapshot._fields['asset_covariance'] is snapshot._fields['raw_diagonal_reference_covariance']
    assert snapshot.asset_covariance is not snapshot.raw_diagonal_reference_covariance


@pytest.mark.parametrize('policy', ['exposure_cluster', 'residual_cluster'])
def test_filtered_fl_block_targets_are_materialized_before_container_removal(policy):
    """FL's original-universe cluster targets survive supplied subset adaptation."""
    filtered = raw_fixture().filter_on_tickers(['z', 'm'])
    snapshot = completed(filtered, policy)
    pd.testing.assert_frame_equal(snapshot.asset_covariance, filtered.get_y_covar(
        residual_type=policy, residual_var_weight=.35, residual_corr_weight=.5), check_exact=True)


def test_alpha_errors_and_prepared_date_mismatch_are_explicit():
    """Missing frequency settings are not guessed; another cutoff cannot be rendered."""
    raw = raw_fixture()
    snapshot = completed(raw)
    with pytest.raises(KeyError, match='QE'):
        fm.estimate_historical_alpha(snapshot, alpha_span={'ME': 24})
    prepared = fm.prepare_snapshot_diagnostics(snapshot)
    other = fm.PreparedSnapshotDiagnostics(prepared.table, cutoff=snapshot.cutoff+pd.Timedelta(days=1),
        alpha_settings=prepared.alpha_settings, alpha_source=prepared.alpha_source, alpha_available=True)
    with pytest.raises(ValueError, match='cutoff and asset order'):
        fm.prepare_portfolio_context(snapshot, diagnostics=other)


def test_plain_opportunity_risk_preserves_supplied_sigma_bits():
    """Consistency tolerances must not replace authoritative supplied covariance."""
    snapshot = completed(raw_fixture(), policy='orthogonal', weight=1.)
    model = fm.FactorOpportunityModel.from_snapshot(snapshot, fm.estimate_historical_alpha(snapshot))
    supplied = model.asset_covariance.copy()
    supplied.iloc[0, 0] = np.nextafter(supplied.iloc[0, 0], np.inf)
    selected = replace(model, asset_covariance=supplied)
    pd.testing.assert_frame_equal(selected.risk_model.covar[model.date], supplied, check_exact=True)
