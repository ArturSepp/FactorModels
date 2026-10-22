"""Offline reference workflows execute against the selected installed/source package."""
from pathlib import Path
import json
import runpy
import socket

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ("minimal_model.py", 1), ("construct_factors.py", 1), ("gls_cma_workflow.py", 1),
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


def test_supplied_bundle_round_trip_preserves_log_risk_and_arithmetic_report(tmp_path, monkeypatch):
    """Load-only inputs round trip; the declared forecast bridge connects the audit."""
    import matplotlib.pyplot as plt
    from factormodels import load_bundle, save_bundle
    from factormodels.reports.cma_diagnostics import _build_pages
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    bundle = namespace["synthetic_bundle"]()
    source = save_bundle(bundle, tmp_path / "supplied")
    loaded = load_bundle(source)
    result = namespace["inspect_fit"](loaded)
    monkeypatch.setenv("FACTORMODELS_OUTPUT_PATH", str(tmp_path / "configured"))
    output = namespace["persist_example"](result, "report")
    assert output == tmp_path / "configured/report"
    assert result["audit"] is not None
    pdf = output / "cma_report.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 1000
    assert (output / "asset_covariance.csv").stat().st_size > 100
    for name in ("loadings.csv", "factor_covariance.csv", "residual_covariance.csv", "result_metadata.json"):
        assert (output / name).stat().st_size > 0
    exported = pd.read_csv(output / "asset_covariance.csv", index_col=0)
    pd.testing.assert_frame_equal(exported, result["snapshot"].asset_covariance, atol=1e-15, rtol=1e-12)
    metadata = json.loads((output / "result_metadata.json").read_text(encoding="utf-8"))
    assert metadata["covariance_units"] == "annual_asset_log_returns_squared"
    assert metadata["report_status"] == result["report_status"]
    for name in ("forecasts.csv", "factor_assumptions.csv", "construction_channels.csv",
                 "assets.csv", "factors.csv", "validation.csv", "gls_alpha_1_factors.csv"):
        assert (output / name).stat().st_size > 0
    audit_metadata = json.loads((output / "audit_metadata.json").read_text(encoding="utf-8"))
    assert audit_metadata["provenance"]["risk_basis"] == "annualized_log_return_covariance"
    assert audit_metadata["provenance"]["return_units"] == "annual_arithmetic_excess_decimal"
    assert audit_metadata["summary"]["accounting_status"] == "pass"
    pages = _build_pages(result["audit"], pages=2)
    try:
        for page in pages:
            text = "\n".join(item.get_text() for item in page.texts)
            assert "Annual log-risk approximation" in text
            assert "Annual arithmetic excess returns" in text
            assert "4 assets / 2 factors" in text
    finally:
        for page in pages:
            plt.close(page)
    forecasts = pd.read_csv(output / "forecasts.csv", index_col=0)
    np.testing.assert_allclose(forecasts.gls_target,
                               forecasts.provider_total-forecasts.cash-forecasts.credited_alpha)
    np.testing.assert_allclose(forecasts.audit_excess_target, forecasts.provider_total-forecasts.cash)
    np.testing.assert_allclose(result["audit"].assets.admitted_alpha, forecasts.credited_alpha)
    with pytest.raises(FileExistsError):
        namespace["persist_example"](result, output)


def test_arithmetic_gls_cma_report_preserves_forecast_channels_and_real_pdf(tmp_path, monkeypatch):
    """A separate arithmetic calibration supplies real table/PDF report coverage."""
    namespace = runpy.run_path(str(ROOT / "examples" / "gls_cma_workflow.py"))
    result = namespace["arithmetic_example"]()
    monkeypatch.setenv("FACTORMODELS_OUTPUT_PATH", str(tmp_path / "configured"))
    output = namespace["persist_example"](result, "arithmetic")
    assert output == tmp_path / "configured/arithmetic"
    pdf = output / "cma_report.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 1000
    for name in ("assets.csv", "factors.csv", "validation.csv", "audit_metadata.json",
                 "forecasts.csv", "asset_covariance.csv", "construction_channels.csv",
                 "gls_alpha_1_factors.csv", "gls_alpha_1_covariance.csv"):
        assert (output / name).stat().st_size > 0
    forecasts = pd.read_csv(output / "forecasts.csv", index_col=0)
    np.testing.assert_allclose(forecasts.gls_target,
                               forecasts.provider_total-forecasts.cash-forecasts.credited_alpha)
    np.testing.assert_allclose(forecasts.audit_excess_target, forecasts.provider_total-forecasts.cash)
    np.testing.assert_allclose(result["audit"].assets.excess_cma, forecasts.audit_excess_target)
    np.testing.assert_allclose(result["audit"].assets.admitted_alpha, forecasts.credited_alpha)
    np.testing.assert_allclose(result["audit"].factors.calibrated_premium, result["views"][1.].premia)
    np.testing.assert_allclose(result["audit"].factors.implied_premium, result["views"][0.].premia)
    metadata = json.loads((output / "audit_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance"]["return_units"] == "annual_arithmetic_excess_decimal"
    assert metadata["provenance"]["historical_return_basis"] == "monthly_simple_excess_decimal"
    assert metadata["provenance"]["cash_rate"] == .03
    assert metadata["summary"]["accounting_status"] == "pass"
    with pytest.raises(FileExistsError):
        namespace["persist_example"](result, output)


def test_arithmetic_cli_does_not_persist_without_output(monkeypatch, capsys):
    """Computing the arithmetic demonstration alone does not render or write files."""
    namespace = runpy.run_path(str(ROOT / "examples" / "gls_cma_workflow.py"))
    def forbidden(*args, **kwargs):
        raise AssertionError("arithmetic example persisted without an explicit output")
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "persist_example", forbidden)
    monkeypatch.setattr("sys.argv", ["gls_cma_workflow.py"])
    main()
    assert "calibrated_premium" in capsys.readouterr().out


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


def test_reference_cli_fit_does_not_persist_without_output(monkeypatch, capsys):
    """An in-memory fit and connected audit never create unrequested report files."""
    namespace = runpy.run_path(str(ROOT / "examples" / "reference_workflows.py"))
    def forbidden(*args, **kwargs):
        raise AssertionError("reference example persisted without explicit output")
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "persist_example", forbidden)
    monkeypatch.setattr("sys.argv", ["reference_workflows.py", "--fit"])
    main()
    assert "available: arithmetic forecast approximation" in capsys.readouterr().out
