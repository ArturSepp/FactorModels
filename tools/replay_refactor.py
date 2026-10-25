"""Capture and verify exact, offline FM refactor evidence outside the checkout.

This is a current-cohort structural regression check, not an independent
mathematical oracle or the historical OP replay. Float/array bits, labelled axes,
dtype and missing-value positions are preserved in plain JSON; no pickle loads.
Capture is exclusive. Verification never changes baseline files. Source identity
is provenance; numerical dependencies, fixture contract and inputs are invariants.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]


def json_bytes(value: object) -> bytes:
    """Encode an already normalized tree canonically, without JSON NaN tokens."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(content: bytes) -> str:
    """Identify exact evidence bytes."""
    return hashlib.sha256(content).hexdigest()


def pack(value: object) -> object:
    """Encode numerical bits and labelled structure; reject unsupported objects.

    DataFrame columns are encoded separately to preserve mixed dtypes. Object
    arrays recurse into supported scalar values rather than persisting pointers.
    Dataclass fields are data, without implementation class/module identities.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return {"kind": "missing", "value": "None" if value is None else str(value)}
    if isinstance(value, Enum):
        return pack(value.value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, pd.DataFrame):
        return {"kind": "frame", "index": pack(value.index), "columns": pack(value.columns),
                "data": [pack(value.iloc[:, i]) for i in range(value.shape[1])], "attrs": pack(value.attrs)}
    if isinstance(value, pd.Series):
        return {"kind": "series", "index": pack(value.index), "name": pack(value.name),
                "dtype": str(value.dtype), "values": pack(value.to_numpy()), "attrs": pack(value.attrs)}
    if isinstance(value, pd.Index):
        return {"kind": "index", "dtype": str(value.dtype), "names": pack(value.names),
                "values": pack(value.to_numpy()), "frequency": getattr(value, "freqstr", None)}
    if isinstance(value, np.ndarray):
        result = {"kind": "array", "dtype": value.dtype.str, "shape": list(value.shape)}
        if value.dtype.hasobject:
            result["items"] = [pack(item) for item in value.flat]
        else:
            result["bytes"] = base64.b64encode(value.tobytes(order="C")).decode("ascii")
        return result
    if isinstance(value, np.generic):
        return {"kind": "numpy_scalar", "value": pack(np.asarray(value))}
    if isinstance(value, float):
        return {"kind": "float64", "bits": struct.pack(">d", value).hex()}
    if isinstance(value, (str, bool, int)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, Mapping):
        # Mapping order is meaningful for dated snapshots and ordered settings.
        return {"kind": "mapping", "items": [[pack(k), pack(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [pack(item) for item in value]}
    if is_dataclass(value) and not isinstance(value, type):
        return {"kind": "record", "fields": {item.name: pack(getattr(value, item.name)) for item in fields(value)}}
    raise TypeError(f"Unsupported evidence type: {type(value).__module__}.{type(value).__name__}")


def assert_exact(actual: object, expected: object, path: str = "root") -> None:
    """Compare normalized trees exactly and identify the first differing field."""
    if type(actual) is not type(expected):
        raise AssertionError(f"{path}: type changed")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: fields changed")
        for key in expected:
            assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length changed")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_exact(a, b, f"{path}[{i}]")
    elif actual != expected:
        raise AssertionError(f"{path}: exact value changed")


def source_manifest() -> dict:
    """Record current source identity without requiring it to stay unchanged."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"root": str(ROOT), "head": git("rev-parse", "HEAD"),
            "dirty_status": git("status", "--porcelain=v1", "--untracked-files=all"),
            "declared_version": project["version"], "installed_version": metadata.version("factormodels"),
            "files": {p: digest((ROOT / p).read_bytes()) for p in sorted(paths) if (ROOT / p).is_file()}}


def cohort() -> dict:
    """Record dependencies, BLAS/solver identity and runtime configuration.

    FM source and distribution version are intentionally excluded: those change
    during the authorized refactor. Numerical dependencies and their source do not.
    """
    import cvxpy as cp
    import factorlasso as fl
    import qis

    versions = {distribution.metadata["Name"].lower(): distribution.version
                for distribution in metadata.distributions()
                if distribution.metadata["Name"].lower() != "factormodels"}
    sources = {}
    for module in (fl, qis):
        folder = Path(module.__file__).resolve().parent
        sources[module.__name__] = {str(p.relative_to(folder)).replace("\\", "/"): digest(p.read_bytes())
                                    for p in sorted(folder.rglob("*.py"))}
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    runtime = io.StringIO()
    with redirect_stdout(runtime):
        np.show_runtime()
    thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONHASHSEED")
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "versions": dict(sorted(versions.items())), "numerical_sources": sources,
            "numpy_build": config.getvalue(), "numpy_runtime": runtime.getvalue(),
            "thread_environment": {name: os.environ.get(name) for name in thread_names},
            "solvers": sorted(cp.installed_solvers())}


def validate_directory(directory: Path) -> Path:
    """Keep generated evidence outside source and OneDrive, including symlinks."""
    target = directory.resolve()
    if target.is_relative_to(ROOT) or any(part.casefold().startswith("onedrive") for part in target.parts):
        raise ValueError("Evidence must be outside the checkout and OneDrive")
    return target


def preflight(mode: str, directory: Path, receipt: Path | None = None) -> None:
    """Reject overwrite/missing evidence before expensive numerical computation."""
    if mode == "capture" and directory.exists():
        raise FileExistsError(f"Refusing to overwrite baseline directory: {directory}")
    if mode == "verify" and not directory.is_dir():
        raise FileNotFoundError(f"Missing baseline: {directory}")
    if receipt is not None:
        if receipt.exists():
            raise FileExistsError(f"Refusing to overwrite receipt: {receipt}")
        if receipt.is_relative_to(directory):
            raise ValueError("Verification receipts must be outside the immutable baseline")


def capture(directory: Path, payload: dict) -> str:
    """Publish one complete payload and its hash manifest, refusing replacement."""
    validate_payload(payload)
    if payload["provenance"]["declared_version"] != "0.2.0":
        raise ValueError("Capture requires the reviewed pre-refactor FM 0.2.0 source")
    content = json_bytes(payload)
    identity = digest(content)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "baseline.json").open("xb") as stream:
        stream.write(content)
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(json_bytes({"schema": SCHEMA, "baseline_sha256": identity}))
    return identity


def load_baseline(directory: Path) -> tuple[dict, str]:
    """Validate evidence completeness, schema and hash before accepting values."""
    manifest = json.loads((directory / "manifest.json").read_bytes())
    content = (directory / "baseline.json").read_bytes()
    identity = digest(content)
    if manifest != {"schema": SCHEMA, "baseline_sha256": identity}:
        raise ValueError("Baseline manifest or payload hash changed")
    payload = json.loads(content)
    validate_payload(payload)
    return payload, identity


def validate_payload(payload: dict) -> None:
    """Reject incomplete evidence even when its integrity hash is self-consistent."""
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "contract", "cohort", "inputs", "outputs", "provenance"}
            or payload["schema"] != SCHEMA):
        raise ValueError("Incomplete or unsupported baseline schema")
    for name in ("inputs", "outputs"):
        value = payload[name]
        if not isinstance(value, dict) or value.get("kind") != "mapping" or not value.get("items"):
            raise ValueError("Empty or malformed evidence is not a baseline")
    if not all(isinstance(payload[key], dict) and payload[key] for key in ("contract", "cohort", "provenance")):
        raise ValueError("Missing fixture, cohort or provenance metadata")


def verify(actual: dict, expected: dict) -> None:
    """Require identical cohort, fixture contract, inputs and economic outputs."""
    for name in ("schema", "contract", "cohort", "inputs", "outputs"):
        assert_exact(actual[name], expected[name], name)


def main() -> None:
    """Run the explicitly selected offline capture or comparison operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    directory = validate_directory(args.directory)
    receipt = validate_directory(args.receipt) if args.receipt is not None else None
    preflight(args.mode, directory, receipt)
    expected, identity = load_baseline(directory) if args.mode == "verify" else (None, None)
    import factormodels as fm
    if Path(fm.__file__).resolve() != ROOT / "src/factormodels/__init__.py":
        raise RuntimeError("Refactor evidence must use the explicit reviewed FM checkout")
    # The case definition is tooling, not an alternate numerical implementation.
    from refactor_cases import CONTRACT, collect
    runtime = cohort()
    if expected is not None:
        assert_exact(runtime, expected["cohort"], "cohort")
        assert_exact(CONTRACT, expected["contract"], "contract")
    inputs, outputs = collect()
    payload = {"schema": SCHEMA, "contract": CONTRACT, "cohort": runtime,
               "inputs": pack(inputs), "outputs": pack(outputs), "provenance": source_manifest()}
    if args.mode == "capture":
        identity = capture(directory, payload)
        print(f"FM_BASELINE_CAPTURED {directory} sha256={identity}", flush=True)
    else:
        verify(payload, expected)
        print(f"FM_EXACT_PASS sha256={identity} cases={len(outputs)}", flush=True)
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        record = {"result": "pass", "mode": args.mode, "baseline_sha256": identity,
                  "cases": list(outputs), "candidate_provenance": payload["provenance"]}
        with receipt.open("xb") as stream:
            stream.write(json_bytes(record))


if __name__ == "__main__":
    main()
