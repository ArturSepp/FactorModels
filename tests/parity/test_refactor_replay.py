"""Acceptance tests for immutable exact FM refactor evidence and root exports."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("replay_refactor", ROOT / "tools/replay_refactor.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def evidence():
    """Disposable evidence, independent of the established external baseline."""
    frame = pd.DataFrame([[1., np.nan], [0., -0.]], index=pd.Index(["a", "b"], name="asset"), columns=["x", "y"])
    return {"schema": replay.SCHEMA, "contract": {"seed": 1}, "cohort": {"solver": "fixed"},
            "inputs": replay.pack({"x": frame.copy()}), "outputs": replay.pack({"y": frame}),
            "provenance": {"declared_version": "0.2.0", "head": "before"}}


def test_capture_roundtrip_and_verification_do_not_write(tmp_path, evidence):
    """Candidate source changes are permitted without touching captured bytes/mtime."""
    directory = tmp_path / "baseline"
    identity = replay.capture(directory, evidence)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    expected, loaded_identity = replay.load_baseline(directory)
    assert loaded_identity == identity
    candidate = deepcopy(evidence)
    candidate["provenance"] = {"declared_version": "0.3.0", "head": "refactored"}
    replay.verify(candidate, expected)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


@pytest.mark.parametrize("defect", ["ulp", "dtype", "row_order", "column_order", "nan_mask", "signed_zero", "axis_name", "shape"])
def test_controlled_output_defects_are_rejected(defect):
    """Every controlled corruption must fail, including a single representable step."""
    expected = pd.DataFrame([[1., np.nan], [0., -0.]], index=pd.Index(["a", "b"], name="asset"), columns=["x", "y"])
    actual = expected.copy()
    if defect == "ulp":
        actual.iloc[0, 0] = np.nextafter(1., np.inf)
        assert actual.iloc[0, 0] != expected.iloc[0, 0]
        assert np.isclose(actual.iloc[0, 0], expected.iloc[0, 0])  # A tolerance gate would miss it.
    elif defect == "dtype":
        actual = actual.astype(np.float32)
    elif defect == "row_order":
        actual = actual.iloc[::-1]
    elif defect == "column_order":
        actual = actual.iloc[:, ::-1]
    elif defect == "nan_mask":
        actual.iloc[0, 1], actual.iloc[1, 1] = actual.iloc[1, 1], actual.iloc[0, 1]
    elif defect == "signed_zero":
        actual.iloc[1, 1] = 0.
    elif defect == "axis_name":
        actual.index.name = "changed"
    else:
        actual = actual.iloc[:1]
    with pytest.raises(AssertionError):
        replay.assert_exact(replay.pack(actual), replay.pack(expected))


@pytest.mark.parametrize("name", ["schema", "contract", "cohort", "inputs", "outputs"])
def test_input_cohort_schema_and_metadata_drift_rejected(evidence, name):
    """No ignored numerical/metadata section can make a changed run appear equal."""
    actual = deepcopy(evidence)
    actual[name] = {"changed": True}
    with pytest.raises(AssertionError):
        replay.verify(actual, evidence)


@pytest.mark.parametrize("defect", ["hash", "missing_file", "missing_field", "schema", "empty", "empty_packed"])
def test_incomplete_and_tampered_baselines_rejected(tmp_path, evidence, defect):
    """Even correctly hashed but incomplete schema is not accepted as evidence."""
    directory = tmp_path / "baseline"
    replay.capture(directory, evidence)
    path = directory / "baseline.json"
    if defect == "missing_file":
        path.unlink()
    elif defect == "hash":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        payload = json.loads(path.read_bytes())
        if defect == "missing_field":
            del payload["outputs"]
        elif defect == "schema":
            payload["schema"] = 999
        elif defect == "empty_packed":
            payload["outputs"] = replay.pack({})
        else:
            payload["outputs"] = {}
        content = replay.json_bytes(payload)
        path.write_bytes(content)
        (directory / "manifest.json").write_bytes(replay.json_bytes(
            {"schema": replay.SCHEMA, "baseline_sha256": replay.digest(content)}))
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.load_baseline(directory)


def test_overwrite_and_missing_evidence_rejected(tmp_path, evidence):
    """Baseline and receipt collisions fail before computation starts."""
    directory, receipt = tmp_path / "baseline", tmp_path / "receipt.json"
    with pytest.raises(FileNotFoundError):
        replay.preflight("verify", directory)
    replay.capture(directory, evidence)
    with pytest.raises(FileExistsError):
        replay.capture(directory, evidence)
    with pytest.raises(FileExistsError):
        replay.preflight("capture", directory)
    receipt.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay.preflight("verify", directory, receipt)
    with pytest.raises(ValueError):
        replay.preflight("verify", directory, directory / "new-receipt.json")
    assert receipt.read_text(encoding="utf-8") == "existing evidence"


@pytest.mark.parametrize("mode", ["capture", "verify"])
def test_cli_preflight_exits_nonzero_for_invalid_destination(tmp_path, mode):
    """Test the real CLI's refusal, not merely a helper in isolation."""
    directory = tmp_path / "baseline"
    if mode == "capture":
        directory.mkdir()
    result = subprocess.run([sys.executable, str(ROOT / "tools/replay_refactor.py"),
                             "--mode", mode, "--directory", str(directory)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert ("Refusing to overwrite" if mode == "capture" else "Missing baseline") in result.stderr
    assert not list(directory.iterdir()) if directory.exists() else True


def test_capture_rejects_post_refactor_source(tmp_path, evidence):
    """A later candidate cannot accidentally become the reviewed old baseline."""
    evidence["provenance"]["declared_version"] = "0.3.0"
    with pytest.raises(ValueError, match="0.2.0"):
        replay.capture(tmp_path / "baseline", evidence)
    assert not (tmp_path / "baseline").exists()


def test_unsupported_objects_and_onedrive_output_are_rejected():
    """Never stringify opaque fitted objects or publish evidence in the checkout."""
    with pytest.raises(TypeError, match="Unsupported"):
        replay.pack(object())
    with pytest.raises(ValueError, match="outside"):
        replay.validate_directory(ROOT / "agents" / "baseline")


def test_frozen_root_api_survives():
    """All reviewed 0.2.0 names remain resolved exports; reviewed additions are allowed."""
    import factormodels as fm
    frozen = json.loads((Path(__file__).with_name("root_api_0_2_0.json")).read_text(encoding="utf-8"))
    assert frozen["version"] == "0.2.0"
    assert len(frozen["names"]) == len(set(frozen["names"]))
    assert set(frozen["names"]) <= set(fm.__all__)
    assert all(hasattr(fm, name) for name in frozen["names"])
