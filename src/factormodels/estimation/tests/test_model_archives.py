"""Lossless native archive and explicit label-view contracts."""
import numpy as np
import pandas as pd
import pytest
import factorlasso as fl
import factormodels as fm

from factormodels.estimation.tests.test_native_results import snapshot


def _equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        _equal(left.attrs, right.attrs)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True)
        _equal(left.attrs, right.attrs)
    elif isinstance(left, pd.Index):
        pd.testing.assert_index_equal(left, right, exact=True)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    elif left is pd.NA or left is pd.NaT:
        assert left is right
    else:
        assert left == right


@pytest.mark.parametrize('rolling', [False, True])
def test_archive_retains_complete_typed_evidence_without_assembly(tmp_path, monkeypatch, rolling):
    """Retain complete typed evidence without fitting or covariance assembly."""
    fitted = snapshot()
    dtype = pd.CategoricalDtype(['unused', 'z', 'a'], ordered=True)
    panel = pd.DataFrame({'label': pd.Categorical(['a', None], dtype=dtype)},
                         index=pd.CategoricalIndex(['z', 'a'], dtype=dtype, name='category'))
    panel.attrs['declaration'] = (pd.NA, pd.NaT, np.array([1., np.nan]))
    fitted = fitted.with_provenance({**fitted.provenance, 'categories': panel})
    model = fm.RollingFactorModel({fitted.cutoff: fitted}) if rolling else fitted
    before = fm.model_to_payload(model)

    def forbidden(*args, **kwargs):
        """Reject any hidden estimation or covariance assembly during reload."""
        raise AssertionError('archive loading must not fit or assemble')

    monkeypatch.setattr(fl.CurrentFactorCovarData, 'get_y_covar', forbidden)
    monkeypatch.setattr(fm.FactorCovarEstimator, 'fit_current', forbidden)
    path = tmp_path / 'model.fmzip'
    digest = fm.write_model_archive(path, model)
    _equal(before, fm.model_to_payload(fm.read_model_archive(path, digest)))
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        fm.read_model_archive(path, digest)


def test_rename_keeps_every_selected_axis_and_original_tree_position(tmp_path):
    """Preserve all selected axes and retained tree identities through renaming."""
    fitted = snapshot().select_assets(['M', 'Z'])
    mapping = {'M': 'Mandate', 'Z': 'Index'}
    renamed = fitted.rename_assets(mapping)
    pd.testing.assert_frame_equal(renamed.asset_covariance,
                                  fitted.asset_covariance.rename(index=mapping, columns=mapping), check_exact=True)
    pd.testing.assert_frame_equal(renamed.residuals, fitted.residuals.rename(columns=mapping), check_exact=True)
    pd.testing.assert_frame_equal(renamed.linkages, fitted.linkages, check_exact=True)
    assert list(renamed.clustering_source_assets['ME']) == ['Index', 'A', 'Mandate']
    pd.testing.assert_frame_equal(renamed.residual_cluster_targets['exposure_cluster'],
        fitted.residual_cluster_targets['exposure_cluster'].rename(index=mapping, columns=mapping), check_exact=True)
    path = tmp_path / 'renamed.fmzip'
    _equal(fm.model_to_payload(renamed), fm.model_to_payload(
        fm.read_model_archive(path, fm.write_model_archive(path, renamed))))
    with pytest.raises(ValueError, match='collides'):
        fitted.rename_assets({'M': 'A'})
    with pytest.raises(ValueError, match='known'):
        fitted.rename_assets({'absent': 'label'})


def test_asset_rename_does_not_rename_observation_cadences():
    """An asset named like a cadence must not alter the native calendar declaration."""
    fitted = snapshot().rename_assets({'M': 'ME'})
    dates = {'ME': fitted.cutoff, 'QE': fitted.cutoff}
    fitted = fitted.with_provenance({**fitted.provenance, 'asset_observation_dates': dates})
    assert fitted.rename_assets({'ME': 'Mandate'}).provenance['asset_observation_dates'] == dates
