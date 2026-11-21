"""Lossless native archive and explicit label-view contracts."""
import hashlib
import json
from zipfile import ZipFile

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
@pytest.mark.parametrize('options', [{}, {'schema': 2}])
def test_archive_retains_complete_typed_evidence_without_assembly(tmp_path, monkeypatch, rolling, options):
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
    monkeypatch.setattr(fm.FactorCovarEstimator, 'fit_rolling', forbidden)
    path = tmp_path / 'model.fmzip'
    digest = fm.write_model_archive(path, model, **options)
    with ZipFile(path) as archive:
        assert json.loads(archive.read('manifest.json'))['schema'] == options.get('schema', 1)
    _equal(before, fm.model_to_payload(fm.read_model_archive(path, digest)))
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        fm.read_model_archive(path, digest)


def test_schema2_compact_frames_preserve_axes_dtypes_and_bits(tmp_path):
    dates = pd.date_range('2024-10-26', periods=3, freq='D', tz='Europe/Zurich', name='as_of')
    bits = np.array([0x8000000000000000, 0, 0x7ff800000000000a, 0x7ff800000000000b,
                     0x3ff0000000000000, 0x4000000000000000], dtype='uint64').reshape(3, 2)
    dense = pd.DataFrame(bits.view('float64'), index=dates, columns=pd.Index(['x', 'x'], name='asset'))
    dense.attrs['source'] = ('test', pd.NA)
    mixed = pd.DataFrame({
        0: pd.array([2**55 + 1, None, 4], dtype='Int64'),
        1: pd.Categorical(['x', None, 'y'], categories=['unused', 'y', 'x'], ordered=True),
        2: dates,
        3: pd.Series(['a', None, 'z'], dtype=object),
    })
    mixed.index = pd.MultiIndex.from_arrays([['a', 'a', 'b'], [1, 1, 2]], names=['asset', 'n'])
    mixed.columns = pd.Index(['metric', 'metric', 'time', 'label'], name='field')
    typed = pd.Series(mixed.iloc[:, 0].array, index=pd.Index(mixed.iloc[:, 0].array, name='id'), name='value')
    frames = dict(dense=dense, mixed=mixed, no_columns=dense.iloc[:, :0], no_rows=dense.iloc[:0], typed_series=typed)
    model = snapshot()
    model = model.with_provenance({**model.provenance, **frames})
    path = tmp_path / 'compact.fmzip'
    restored = fm.read_model_archive(path, fm.write_model_archive(path, model, schema=2))
    for name, frame in frames.items():
        _equal(frame, restored.provenance[name])
    np.testing.assert_array_equal(restored.provenance['dense'].to_numpy().view('uint64'), bits)
    with ZipFile(path) as archive:
        manifest = archive.read('manifest.json')
        assert b'"frame_block"' in manifest and b'"frame_columns"' in manifest
        assert b'"type": "frame",' not in manifest


@pytest.mark.parametrize('unsupported', [0, 3, True, 1.0, '2'])
def test_schema_validation_preserves_existing_archive(tmp_path, unsupported):
    path = tmp_path / 'model.fmzip'
    model = snapshot()
    digest = fm.write_model_archive(path, model)
    with pytest.raises(ValueError, match='schema'):
        fm.write_model_archive(path, model, schema=unsupported)
    assert fm.read_model_archive(path, digest).cutoff == model.cutoff
    with ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(dict(schema=unsupported, payload={})))
    with pytest.raises(ValueError, match='storage schema'):
        fm.read_model_archive(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize(('kind', 'schema', 'message'), [
    ('frame_block', 1, 'unsupported fitted payload type'),
    ('frame_block', 2, 'block shape'),
    ('frame_columns', 2, 'column count'),
])
def test_reader_validates_frame_layout_before_model_construction(tmp_path, kind, schema, message):
    from io import BytesIO

    axis = dict(type='rangeindex', start=0, stop=2, step=1, name=None)
    frame = dict(type=kind, index=axis, columns=axis, attrs=dict(type='mapping', items=[]))
    buffer = BytesIO()
    np.save(buffer, np.ones((2, 1)), allow_pickle=False)
    if kind == 'frame_block':
        frame['block'] = dict(type='array', file='arrays/0.npy')
    else:
        frame['data'] = []
    path = tmp_path / 'shape.fmzip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('arrays/0.npy', buffer.getvalue())
        archive.writestr('manifest.json', json.dumps(dict(schema=schema, payload=frame)))
    with pytest.raises(ValueError, match=message):
        fm.read_model_archive(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_failed_write_preserves_previous_model_and_cleans_staging(tmp_path):
    model = snapshot()
    path = tmp_path / 'fitted.fmzip'
    digest = fm.write_model_archive(path, model)
    invalid = model.with_provenance({'unsupported': object()})
    with pytest.raises(TypeError, match='unsupported fitted metadata'):
        fm.write_model_archive(path, invalid, schema=2)
    assert list(tmp_path.iterdir()) == [path]
    _equal(fm.model_to_payload(model), fm.model_to_payload(fm.read_model_archive(path, digest)))


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
