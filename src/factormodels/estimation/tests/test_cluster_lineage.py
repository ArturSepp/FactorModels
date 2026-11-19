"""Native lineage preparation preserves the existing statistical outputs exactly."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, analyze_cluster_lineage
from factorlasso.diagnostics._lineage import _snapshot_fingerprints

import factormodels as fm


def _history():
    """Supply dated clusters, unequal asset risks and changing fitted loadings."""
    assets = pd.Index(['a', 'b', 'c'])
    return RollingFactorCovarData({date: CurrentFactorCovarData(
        pd.DataFrame([[.04, .003], [.003, .01]], index=['equity', 'rates'], columns=['equity', 'rates']),
        pd.DataFrame([[1.+i*.01, .1], [.8, .3], [.1, .9]], index=assets, columns=['equity', 'rates']),
        pd.DataFrame({'residual_var': [.01, .02, .015]}, index=assets),
        estimation_date=date, clusters=pd.Series(['ME:1', 'ME:1', 'ME:2'], index=assets))
        for i, date in enumerate(pd.date_range('2024-01-31', periods=3, freq='ME'))})


@pytest.mark.parametrize('weighting', ['equal', 'inv_vol'])
def test_lineage_and_fingerprints_match_statistical_reference_exactly(weighting):
    """All report panels, parameters and fingerprint fields match FL without refitting."""
    raw = _history()
    native = fm.snapshots_from_factorlasso(raw, residual_type='orthogonal', residual_var_weight=.35)
    expected = analyze_cluster_lineage(raw, weighting=weighting)
    actual = fm.prepare_cluster_lineage(native, weighting=weighting)
    for name in ('classification', 'lineage', 'transitions', 'relabel', 'factor_covar'):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
    assert actual.params == expected.params
    pd.testing.assert_frame_equal(actual.to_label_panel(), expected.to_label_panel(), check_exact=True)
    pd.testing.assert_frame_equal(actual.to_membership_panel(), expected.to_membership_panel(), check_exact=True)
    for date, snapshot in native.snapshots.items():
        fields, factors = fm.prepare_cluster_fingerprints(snapshot, weighting=weighting)
        reference, names = _snapshot_fingerprints(raw.data[date], weighting=weighting)
        assert factors == names and fields.keys() == reference.keys()
        for label, values in fields.items():
            for name, expected_value in asdict(reference[label]).items():
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_array_equal(values[name], expected_value)
                else:
                    assert values[name] == expected_value
        fields[next(iter(fields))]['beta'][:] = 999.
        pd.testing.assert_frame_equal(snapshot.loadings, raw.data[date].y_betas, check_exact=True)


def test_lineage_preparation_rejects_raw_models_and_unknown_weighting():
    """The native preparation boundary never silently accepts raw fitted containers."""
    raw = _history()
    with pytest.raises(TypeError, match='RollingFactorModel'):
        fm.prepare_cluster_lineage(raw)
    with pytest.raises(TypeError, match='FactorModelSnapshot'):
        fm.prepare_cluster_fingerprints(next(iter(raw.data.values())))
    native = fm.snapshots_from_factorlasso(raw, residual_type='orthogonal')
    with pytest.raises(ValueError, match='weighting'):
        fm.prepare_cluster_fingerprints(next(iter(native.snapshots.values())), weighting='unknown')
