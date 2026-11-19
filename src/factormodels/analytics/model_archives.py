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

from typing import Any
from enum import Enum
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from qis import ReturnTypes
from factormodels.analytics.models import FactorModelSnapshot, RollingFactorModel
from factormodels.analytics.model_payloads import model_from_payload, model_to_payload

__all__ = ["write_model_archive", "read_model_archive"]


def _encode(value: Any, archive: ZipFile, arrays: list) -> Any:
    """Encode only plain FM field types, retaining axes, dtypes and missing masks."""
    def encode(item: Any) -> Any:
        return _encode(item, archive, arrays)
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
        return {'type': 'frame', 'index': encode(value.index), 'columns': encode(value.columns),
                'data': [encode(value.iloc[:, i]) for i in range(len(value.columns))],
                'attrs': encode(value.attrs)}
    if isinstance(value, pd.Series):
        return {'type': 'series', 'index': encode(value.index), 'name': encode(value.name),
                'dtype': (encode(value.dtype) if isinstance(value.dtype, pd.CategoricalDtype)
                          else str(value.dtype)),
                'values': encode(value.to_numpy()), 'attrs': encode(value.attrs)}
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
        return {'type': 'index', 'values': encode(value.to_numpy()),
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


def _decode(value: Any, archive: ZipFile) -> Any:
    """Read the fixed type vocabulary without dynamic imports or unpickling."""
    if not isinstance(value, dict):
        return value
    def decode(item: Any) -> Any:
        return _decode(item, archive)
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
    if kind == 'frame':
        columns = [decode(v) for v in value['data']]
        result = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=decode(value['index']))
        result.index, result.columns = decode(value['index']), decode(value['columns'])
        result.attrs = decode(value['attrs'])
        return result
    raise ValueError(f'unsupported fitted payload type {kind!r}')


def write_model_archive(path: Path | str, model: FactorModelSnapshot | RollingFactorModel) -> str:
    """Atomically persist complete native state without executable object serialization.

    Args:
        path: Destination for the schema-1 ZIP archive; parent directories are created.
        model: Completed snapshot or rolling result, including all fitted evidence.

    Returns:
        SHA-256 digest to retain alongside the application artifact.
    """
    path = Path(path)
    if not isinstance(model, (FactorModelSnapshot, RollingFactorModel)):
        raise TypeError('model must be FactorModelSnapshot or RollingFactorModel')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as archive:
        tree = _encode(model_to_payload(model), archive, [])
        archive.writestr('manifest.json', json.dumps({'schema': 1, 'payload': tree}, allow_nan=False))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_model_archive(path: Path | str, expected_sha256: str) -> FactorModelSnapshot | RollingFactorModel:
    """Verify and reconstruct a complete native result without fitting or assembly.

    Args:
        path: Schema-1 archive written by :func:`write_model_archive`.
        expected_sha256: Required digest recorded with the caller's artifact.

    Returns:
        Validated snapshot or rolling result retaining exact typed fitted fields.

    Raises:
        ValueError: If the digest, archive schema or model payload is invalid.
    """
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('fitted factor model checksum mismatch')
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if set(manifest) != {'schema', 'payload'} or manifest['schema'] != 1:
            raise ValueError('unsupported fitted factor model storage schema')
        model = model_from_payload(_decode(manifest['payload'], archive))
    return model
