"""The README is an executable offline path from construction to dated covariance."""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import re
import socket

import numpy as np
import pandas as pd
import pytest


def _normalised_output(output):
    """Ignore display whitespace and QIS's explicitly identified elapsed-time line."""
    output = re.sub(
        r"^Finished 'compute_returns_in_reference_ccy' in [0-9.]+ secs\s*$",
        "", output, flags=re.MULTILINE,
    )
    return " ".join(output.split())


def _execute_readme(readme):
    """Execute every Python fence in order and compare the two displayed outputs."""
    blocks = re.findall(r"^```(python|text)[ \t]*\n(.*?)^```[ \t]*$", readme, re.M | re.S)
    assert sum(kind == "python" for kind, _ in blocks) == 6, "Expected six Python fences"
    assert sum(kind == "text" for kind, _ in blocks) == 2, "Expected two output fences"
    namespace = {"__name__": "__readme__"}
    output = ""
    for number, (kind, body) in enumerate(blocks, 1):
        if kind == "python":
            captured = StringIO()
            with redirect_stdout(captured):
                exec(compile(body, f"README.md:fence-{number}", "exec"), namespace)
            output = captured.getvalue()
        else:
            assert _normalised_output(output) == _normalised_output(body), "README output differs"
    return namespace


def test_readme_python_fences_execute_offline_with_causal_covariance(monkeypatch, tmp_path):
    """Check literal examples, total risk and their historical-information promises."""
    def reject_network(*args, **kwargs):
        """Fail an accidental acquisition in an advertised offline quickstart."""
        raise AssertionError("README examples must not access the network")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    namespace = _execute_readme(readme)
    assert not list(tmp_path.iterdir()), "README examples must not write output files"
    for result in (namespace["model"], namespace["snapshot"]):
        total = result.loadings @ result.factor_covariance @ result.loadings.T
        total += result.residual_covariance
        pd.testing.assert_frame_equal(result.asset_covariance, total, rtol=1e-11, atol=1e-12)

    estimator = namespace["estimator"]
    factors = namespace["factor_prices"]
    returns = namespace["returns_by_cadence"]
    supplied = namespace["supplied"]
    snapshot = namespace["snapshot"]
    assert snapshot.cutoff == pd.Timestamp("2021-12-31")
    assert supplied.observation_date <= supplied.available_date <= snapshot.cutoff
    assert snapshot.provenance["factor_covariance"] == "supplied"
    assert snapshot.provenance["covariance_observation_date"] == supplied.observation_date
    assert snapshot.provenance["covariance_available_date"] == supplied.available_date
    # A date label alone does not prove that the matrix used only past information.
    historical = estimator.fit_current(factors, returns, cutoff=supplied.observation_date)
    pd.testing.assert_frame_equal(supplied.matrix, historical.factor_covariance)
    pd.testing.assert_frame_equal(snapshot.factor_covariance, supplied.matrix)

    rolling = namespace["rolling"].snapshots
    assert len(rolling) > 1 and max(rolling) == snapshot.cutoff
    for date, result in rolling.items():
        assert result.cutoff == date
        assert result.provenance["factor_observation_date"] <= date
        assert all(value <= date for value in result.provenance["asset_observation_dates"].values())
    # Both endpoints must agree with separately truncated histories.
    for date in (min(rolling), max(rolling)):
        reference = estimator.fit_current(
            factors.loc[:date], {freq: values.loc[:date] for freq, values in returns.items()},
            cutoff=date,
        )
        np.testing.assert_allclose(rolling[date].loadings, reference.loadings, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            rolling[date].asset_covariance, reference.asset_covariance, rtol=1e-11, atol=1e-12,
        )
    late = replace(supplied, available_date=snapshot.cutoff + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="not available at cutoff"):
        estimator.fit_current(factors, returns, cutoff=snapshot.cutoff, factor_covariance=late)
