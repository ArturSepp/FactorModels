"""Repository teaching inputs preserve their fixed observations and isolation."""
import builtins
import io
from pathlib import Path
import runpy
import socket

import numpy as np
import pandas as pd
import factormodels as tf
import pytest

synthetic_prices = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "examples/reference_workflows.py"))["synthetic_prices"]


def test_synthetic_factory_is_reproducible_and_returns_fresh_inputs():
    """Each call owns fresh price tables with the same deterministic observations."""
    first, second = synthetic_prices(), synthetic_prices()
    for position, (original, independent) in enumerate(zip(first, second)):
        pd.testing.assert_frame_equal(original, independent, check_exact=True)
        before = independent.copy()
        original.iloc[0, 0] = -999.
        original.index.name = "changed index"
        pd.testing.assert_frame_equal(independent, before, check_exact=True)
        pd.testing.assert_frame_equal(synthetic_prices()[position], before, check_exact=True)
    assert second[0].shape == (121, 2)
    assert second[1].shape == (121, 4)
    assert second[0].index[[0, -1]].tolist() == [pd.Timestamp("2014-01-31"), pd.Timestamp("2024-01-31")]


@pytest.mark.parametrize("position", [0, 1])
def test_synthetic_tables_have_independent_index_metadata(position):
    """Renaming either table's index must leave the other table unchanged."""
    tables = synthetic_prices()
    sibling = tables[1 - position]
    before = sibling.copy(deep=True)
    tables[position].index.name = "changed index"
    pd.testing.assert_frame_equal(sibling, before, check_exact=True)


def test_synthetic_factory_needs_no_fit_network_or_files_and_preserves_global_rng(monkeypatch):
    """Input creation is an isolated in-memory action, with a local random generator."""
    before = np.random.get_state()

    def forbidden(*args, **kwargs):
        """Reject hidden computation, acquisition or filesystem activity."""
        raise AssertionError("synthetic inputs must be produced only in memory")

    with monkeypatch.context() as patches:
        patches.setattr(tf.FactorCovarEstimator, "fit_current", forbidden)
        patches.setattr(tf.FactorCovarEstimator, "fit_rolling", forbidden)
        patches.setattr(socket.socket, "connect", forbidden)
        patches.setattr(socket, "create_connection", forbidden)
        patches.setattr(builtins, "open", forbidden)
        patches.setattr(io, "open", forbidden)
        patches.setattr(pd.DataFrame, "to_csv", forbidden)
        factors, _ = synthetic_prices()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert factors.shape == (121, 2)
