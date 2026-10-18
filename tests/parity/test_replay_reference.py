"""Controlled counterexamples for the immutable numerical replay contract."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("replay_reference", Path(__file__).parents[2]/"tools/replay_reference.py")
replay = module_from_spec(spec)
spec.loader.exec_module(replay)


def test_covariance_perturbation_is_rejected():
    """The recorded tolerance detects a deliberate economically material defect."""
    expected = {"covariance": [[.04, .01], [.01, .09]]}
    replay.check(expected, expected, atol=1e-10, rtol=1e-8)
    bad = {"covariance": [[.041, .01], [.01, .09]]}
    with pytest.raises(AssertionError):
        replay.check(bad, expected, atol=1e-10, rtol=1e-8)


def test_axis_change_is_rejected_even_with_identical_numbers():
    """Equal values do not make a reordered asset universe interchangeable."""
    expected = {"index": ["asset-a", "asset-b"], "values": [.1,.2]}
    with pytest.raises(AssertionError, match="index"):
        replay.check({"index":["asset-b","asset-a"],"values":[.1,.2]}, expected,
                     atol=1e-10, rtol=1e-8)


def test_missing_component_is_rejected():
    """An empty output must never report a passing numerical comparison."""
    with pytest.raises(AssertionError, match="keys changed"):
        replay.check({}, {"rolling":{"2025-01-31":.04}}, atol=1e-10, rtol=1e-8)
