"""Portable lossless file storage for FM's versioned plain-field result payload.

Purpose: Persist complete fitted state without precision or metadata loss.
Inputs: A completed native result, an archive path and its expected digest.
Outputs: A checksummed archive or a validated native result without refitting.
Units: Every original return coordinate and annualisation declaration is retained.
Conditioning: Storage performs no estimation, covariance assembly or numeric casting.
Ownership: FM owns this shared codec; applications own paths and publication policy.

Only declared scalar/table types are decoded. Numeric arrays use NumPy files
with pickle disabled; no import name or executable object is read from storage.
Applications own paths and workbook policy; FM validates every selected policy.
The fixed archive vocabulary is shared by independent model consumers.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
from enum import Enum
from io import BytesIO
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from qis import ReturnTypes
from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel
from factormodels.analytics.model_payloads import model_from_payload, model_to_payload

__all__ = ["write_model_archive", "read_model_archive"]


def _pandas_values(value: pd.Series | pd.Index, schema: int) -> np.ndarray:
    """Keep extension scalars and missing markers distinct in compact archives."""
    if schema == 2 and isinstance(value.dtype, pd.api.extensions.ExtensionDtype):
        return value.to_numpy(dtype=object)
    return value.to_numpy()


def _encode_frame(frame: pd.DataFrame, encode: Callable[[Any], Any]) -> dict:
    """Store shared axes once and use a block only for an exact NumPy dtype match."""
    axes = dict(index=encode(frame.index), columns=encode(frame.columns), attrs=encode(frame.attrs))
    dtypes = list(frame.dtypes)
    if dtypes and isinstance(dtypes[0], np.dtype) and dtypes[0].kind in 'biufcmM':
        if all(dtype == dtypes[0] for dtype in dtypes):
            block = frame.to_numpy(copy=False)
            if block.dtype == dtypes[0]:
                return dict(type='frame_block', block=encode(block), **axes)

    columns = []
    for position, dtype in enumerate(dtypes):
        series = frame.iloc[:, position]
        stored_dtype = encode(dtype) if isinstance(dtype, pd.CategoricalDtype) else str(dtype)
        columns.append(dict(dtype=stored_dtype, values=encode(_pandas_values(series, 2))))
    return dict(type='frame_columns', data=columns, **axes)


def _encode(value: Any, archive: ZipFile, arrays: list, schema: int = 1) -> Any:
    """Encode only plain FM field types, retaining axes, dtypes and missing masks."""
    def encode(item: Any) -> Any:
        return _encode(item, archive, arrays, schema)
    if value is pd.NA:
        return {'type': 'NA'}
    if value is pd.NaT:
        return {'type': 'NaT'}
    if isinstance(value, Enum):
        if not isinstance(value, ReturnTypes):
            raise TypeError(f'unsupported fitted metadata enum {type(value)}')
        return {'type': 'ReturnTypes', 'name': value.name}
    if isinstance(value, pd.Timestamp):
        return {'type': 'timestamp', 'value': value.isoformat()}
    if isinstance(value, pd.CategoricalDtype):
        return {'type': 'categorical_dtype', 'categories': encode(value.categories),
                'ordered': value.ordered}
    if isinstance(value, np.generic):
        return {'type': 'scalar', 'dtype': str(value.dtype), 'value': encode(value.item())}
    if isinstance(value, float) and not np.isfinite(value):
        return {'type': 'float', 'value': str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, pd.DataFrame):
        if schema == 2:
            return _encode_frame(value, encode)
        return {'type': 'frame', 'index': encode(value.index), 'columns': encode(value.columns),
                'data': [encode(value.iloc[:, i]) for i in range(len(value.columns))],
                'attrs': encode(value.attrs)}
    if isinstance(value, pd.Series):
        return {'type': 'series', 'index': encode(value.index), 'name': encode(value.name),
                'dtype': (encode(value.dtype) if isinstance(value.dtype, pd.CategoricalDtype)
                          else str(value.dtype)),
                'values': encode(_pandas_values(value, schema)), 'attrs': encode(value.attrs)}
    if isinstance(value, pd.MultiIndex):
        return {'type': 'multiindex', 'levels': [encode(v) for v in value.levels],
                'codes': [encode(v) for v in value.codes], 'names': encode(value.names)}
    if isinstance(value, pd.RangeIndex):
        return {'type': 'rangeindex', 'start': value.start, 'stop': value.stop,
                'step': value.step, 'name': encode(value.name)}
    if isinstance(value, pd.DatetimeIndex):
        return {'type': 'datetimeindex', 'values': encode(value.to_numpy()), 'dtype': str(value.dtype),
                'freq': value.freqstr, 'name': encode(value.name)}
    if isinstance(value, pd.Index):
        if type(value) not in (pd.Index, pd.CategoricalIndex):
            raise TypeError(f'unsupported fitted index type {type(value)}')
        return {'type': 'index', 'values': encode(_pandas_values(value, schema)),
                'dtype': (encode(value.dtype) if isinstance(value.dtype, pd.CategoricalDtype)
                          else str(value.dtype)),
                'name': encode(value.name)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {'type': 'objects', 'shape': list(value.shape),
                    'values': [encode(v) for v in value.ravel()]}
        name = f'arrays/{len(arrays)}.npy'
        arrays.append(name)
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        archive.writestr(name, buffer.getvalue())
        return {'type': 'array', 'file': name}
    if isinstance(value, dict):
        return {'type': 'mapping', 'items': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'type': 'tuple' if isinstance(value, tuple) else 'list', 'items': [encode(v) for v in value]}
    raise TypeError(f'unsupported fitted metadata type {type(value)}')


def _decode_frame(node: dict, decode: Callable[[Any], Any]) -> pd.DataFrame:
    """Restore a schema-2 frame by position, including empty or repeated axes."""
    index, columns = decode(node['index']), decode(node['columns'])
    if not isinstance(index, pd.Index) or not isinstance(columns, pd.Index):
        raise ValueError('invalid fitted frame axes')
    if node['type'] == 'frame_block':
        block = decode(node['block'])
        if not isinstance(block, np.ndarray) or block.dtype.kind not in 'biufcmM':
            raise ValueError('invalid fitted frame block dtype')
        if block.shape != (len(index), len(columns)):
            raise ValueError('invalid fitted frame block shape')
        result = pd.DataFrame(block, index=index, columns=columns, copy=False)
    else:
        if len(node['data']) != len(columns):
            raise ValueError('invalid fitted frame column count')
        series = []
        for column in node['data']:
            values = decode(column['values'])
            if not isinstance(values, np.ndarray) or values.shape != (len(index),):
                raise ValueError('invalid fitted frame column shape')
            series.append(pd.Series(values, index=index, dtype=decode(column['dtype'])))
        result = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
        result.columns = columns
    result.attrs = decode(node['attrs'])
    return result


def _decode(value: Any, archive: ZipFile, schema: int = 1) -> Any:
    """Read the fixed type vocabulary without dynamic imports or unpickling."""
    if not isinstance(value, dict):
        return value
    def decode(item: Any) -> Any:
        return _decode(item, archive, schema)
    kind = value['type']
    if kind == 'NA':
        return pd.NA
    if kind == 'NaT':
        return pd.NaT
    if kind == 'ReturnTypes':
        return ReturnTypes[value['name']]
    if kind == 'timestamp':
        return pd.Timestamp(value['value'])
    if kind == 'categorical_dtype':
        return pd.CategoricalDtype(decode(value['categories']), ordered=value['ordered'])
    if kind == 'float':
        return float(value['value'])
    if kind == 'scalar':
        return np.asarray(decode(value['value']), dtype=value['dtype'])[()]
    if kind == 'array':
        return np.load(BytesIO(archive.read(value['file'])), allow_pickle=False)
    if kind == 'objects':
        result = np.empty(len(value['values']), dtype=object)
        result[:] = [decode(v) for v in value['values']]
        return result.reshape(value['shape'])
    if kind == 'mapping':
        return {decode(k): decode(v) for k, v in value['items']}
    if kind in ('list', 'tuple'):
        result = [decode(v) for v in value['items']]
        return tuple(result) if kind == 'tuple' else result
    if kind == 'rangeindex':
        return pd.RangeIndex(value['start'], value['stop'], value['step'], name=decode(value['name']))
    if kind == 'multiindex':
        return pd.MultiIndex(levels=[decode(v) for v in value['levels']],
                             codes=[decode(v) for v in value['codes']], names=decode(value['names']))
    if kind == 'index':
        return pd.Index(decode(value['values']), dtype=decode(value['dtype']), name=decode(value['name']))
    if kind == 'datetimeindex':
        return pd.DatetimeIndex(decode(value['values']), dtype=value['dtype'],
                                freq=value['freq'], name=decode(value['name']))
    if kind == 'series':
        result = pd.Series(decode(value['values']), index=decode(value['index']),
                           name=decode(value['name']), dtype=decode(value['dtype']))
        result.attrs = decode(value['attrs'])
        return result
    if schema == 2 and kind in ('frame_block', 'frame_columns'):
        return _decode_frame(value, decode)
    if schema == 1 and kind == 'frame':
        columns = [decode(v) for v in value['data']]
        result = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=decode(value['index']))
        result.index, result.columns = decode(value['index']), decode(value['columns'])
        result.attrs = decode(value['attrs'])
        return result
    raise ValueError(f'unsupported fitted payload type {kind!r}')


def write_model_archive(path: Path | str, model: FactorModelSnapshot | RollingFactorModel, *, schema: int = 1) -> str:
    """Atomically persist complete native state without executable object serialization.

    Args:
        path: Destination archive. Parent directories are created as needed.
        model: Completed snapshot or rolling result, including all fitted evidence.
        schema: Archive format. The default 1 is readable by FM 0.6.0. Select 2
            for compact frames after all consumers support that format.

    Returns:
        SHA-256 digest to retain alongside the application artifact.
    """
    path = Path(path)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError('unsupported fitted factor model storage schema')
    if not isinstance(model, (FactorModelSnapshot, RollingFactorModel)):
        raise TypeError('model must be FactorModelSnapshot or RollingFactorModel')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x+b') as stream:
            with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as archive:
                tree = _encode(model_to_payload(model), archive, [], schema)
                archive.writestr('manifest.json', json.dumps({'schema': schema, 'payload': tree}, allow_nan=False))
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        temporary.replace(path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def read_model_archive(path: Path | str, expected_sha256: str) -> FactorModelSnapshot | RollingFactorModel:
    """Verify and reconstruct a complete native result without fitting or assembly.

    Args:
        path: A schema 1 or schema 2 archive written by :func:`write_model_archive`.
        expected_sha256: Required digest recorded with the caller's artifact.

    Returns:
        Validated snapshot or rolling result retaining exact typed fitted fields.

    Raises:
        ValueError: If the digest, archive schema or model payload is invalid.
    """
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('fitted factor model checksum mismatch')
        stream.seek(0)
        with ZipFile(stream) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            if not isinstance(manifest, dict) or set(manifest) != {'schema', 'payload'}:
                raise ValueError('invalid fitted factor model archive manifest')
            schema = manifest['schema']
            if type(schema) is not int or schema not in (1, 2):
                raise ValueError('unsupported fitted factor model storage schema')
            return model_from_payload(_decode(manifest['payload'], archive, schema))
