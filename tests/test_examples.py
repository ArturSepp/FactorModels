"""Offline reference workflows execute against the selected installed/source package."""
from pathlib import Path
import runpy
import socket

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ("construct_factors.py", 1), ("gls_factor_premia.py", 1),
    ("reference_workflows.py", 1), ("reference_workflows.py", 2),
    ("methods/factor_covariance_hcgl.py", 1), ("methods/gls_factor_premia.py", 1),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Provider access is an error in every offline reference path."""
    def denied(*args, **kwargs):
        raise AssertionError("offline examples must not access the network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.mark.parametrize("relative,case", EXAMPLES)
def test_offline_reference(relative, case):
    """Every canonical offline example has an actual completion assertion."""
    namespace = runpy.run_path(str(ROOT / "examples" / relative))
    namespace["run_local_test"](case)


def test_supplied_bundle_round_trip_and_report(tmp_path):
    """Load-only inputs round trip; explicit fit produces tables and a genuine PDF."""
    from tradablefactors import load_bundle, save_bundle
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    source = save_bundle(bundle, tmp_path / "supplied")
    loaded = load_bundle(source)
    result = namespace["inspect_fit"](loaded)
    output = namespace["persist_example"](result, tmp_path / "report")
    assert (output / "cma_report.pdf").read_bytes().startswith(b"%PDF-")
    assert (output / "cma_report.pdf").stat().st_size > 1000
    assert (output / "asset_covariance.csv").stat().st_size > 100
    for name in ("assets.csv", "factors.csv", "validation.csv", "audit_metadata.json", "result_metadata.json"):
        assert (output / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        namespace["persist_example"](result, output)


def test_reference_cli_defaults_to_load_only(monkeypatch, capsys):
    """No option triggers a hidden fit, acquisition or persistence."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    def forbidden(*args, **kwargs):
        raise AssertionError("load-only entry point attempted another action")
    main = namespace["main"]
    for name in ("fit_bundle", "inspect_fit", "persist_example", "acquire_proxy_prices"):
        monkeypatch.setitem(main.__globals__, name, forbidden)
    monkeypatch.setattr("sys.argv", ["reference_workflows.py"])
    main()
    assert "Loaded 121 factor observations" in capsys.readouterr().out
