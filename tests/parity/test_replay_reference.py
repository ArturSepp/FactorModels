"""Controlled counterexamples for the immutable numerical replay contract."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
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


@pytest.mark.parametrize("corrupt", [False, True])
def test_consumer_construction_executes_rosaa_boundary_and_rejects_corruption(monkeypatch, corrupt):
    """A direct TF fallback would miss the deliberately corrupted Rosaa output."""
    from tradablefactors.construction import build_factor_panel

    module = ModuleType("ramen.quant_model_mvp.rosaa.data.factors.panel_io")
    module.__file__ = __file__
    module.load_base_futures_prices = lambda **kwargs: pytest.fail("Real input IO must be isolated")
    module.save_factor_panel = lambda **kwargs: pytest.fail("Accepted output must not be written")
    calls = []

    def update_factor_panel(*, local_path, factor_model, rebalancing_freq, vol_span):
        calls.append((factor_model.name, rebalancing_freq, vol_span))
        panel = build_factor_panel(factor_model, module.load_base_futures_prices(local_path=local_path),
                                   rebalancing_freq=rebalancing_freq, vol_span=vol_span)
        if corrupt:
            panel.iloc[-1, 0] += 1.0
        module.save_factor_panel(factors=panel, factor_model=factor_model, local_path=local_path)

    module.update_factor_panel = update_factor_panel
    parent = ModuleType("ramen.quant_model_mvp.rosaa.data.factors")
    parent.panel_io = module
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    if corrupt:
        with pytest.raises(AssertionError):
            replay.construction("consumer")
    else:
        receipt = {}
        actual = replay.construction("consumer", receipt)
        expected = replay.construction("tradablefactors")
        replay.check(replay.labelled(actual), replay.labelled(expected), atol=1e-10, rtol=1e-8)
        assert receipt["construction"]["entry_point"] == "panel_io.update_factor_panel"
    assert calls == [("Frozen quarterly basket", "QE", None)]


def paper_adapters(monkeypatch, *, corrupt_faj=False, corrupt_matf=False):
    """Supply executable consumer seams without private source/data in public tests."""
    from tradablefactors.cma.diagnostics import CmaDiagnosticInputs, compute_cma_diagnostics
    betas = pd.DataFrame([[1., 0.], [0., 1.], [.5, .6], [1., -.2]],
                         index=list("abcd"), columns=["equity", "rates"])
    residual = pd.Series([.01, .02, .03, .04], index=betas.index)
    covariance = pd.DataFrame([[.04, .005], [.005, .01]], index=betas.columns, columns=betas.columns)
    premia = pd.Series([.05, .02], index=betas.columns)
    mean = betas @ premia + pd.Series([.012, .015, -.011, .004], index=betas.index)
    calls = []

    def load_production_inputs(snapshot):
        calls.append(("faj", snapshot))
        return {"tickers": list(betas.index), "factors": list(betas.columns), "beta": betas.to_numpy(),
                "D": residual.to_numpy() * (1.1 if corrupt_faj else 1.), "Sig_F": covariance.to_numpy(),
                "lam": premia.to_numpy(), "mu_excess_full": mean.to_numpy(),
                "inputs": SimpleNamespace(manifest={"source": "test-saved-fit"})}

    matf = SimpleNamespace(compute_cma_diagnostics=compute_cma_diagnostics)

    def diagnose_paper_snapshot(snapshot):
        calls.append(("matf", snapshot))
        result = matf.compute_cma_diagnostics(CmaDiagnosticInputs(
            betas=betas, residual_variances=residual, excess_cma=mean,
            factor_covariance=covariance, factor_premia=premia))
        if corrupt_matf:
            result.assets.iloc[0, result.assets.columns.get_loc("residual")] += .01
        return result

    matf.diagnose_paper_snapshot = diagnose_paper_snapshot

    @contextmanager
    def module(root, paper, name):
        yield SimpleNamespace(load_production_inputs=load_production_inputs) if name == "paper_inputs" else matf

    monkeypatch.setattr(replay, "paper_module", module)
    monkeypatch.setattr(replay, "consumer_source_identity", lambda root: {"test.py": "test-identity"})
    return calls


def test_consumer_papers_call_both_real_entry_points_with_explicit_saved_cuts(monkeypatch, tmp_path):
    """The consumer path must observe both adapters, preserve inputs and return matching economics."""
    calls = paper_adapters(monkeypatch)
    mapping = {"faj_q3": tmp_path / "faj", "matf_current": tmp_path / "matf"}
    receipt = {}
    result = replay.paper_results("consumer", mapping, consumer_root=tmp_path, receipt=receipt)
    expected = replay.diagnose("tradablefactors")
    for value in result.values():
        replay.check(replay.labelled(value), replay.labelled(expected), atol=1e-10, rtol=1e-8)
    assert calls == [("faj", str(mapping["faj_q3"].resolve())),
                     ("matf", str(mapping["matf_current"].resolve()))]
    assert receipt["matf_current"]["entry_point"].endswith("diagnose_paper_snapshot")


@pytest.mark.parametrize("consumer", ["faj", "matf"])
def test_corrupted_consumer_cannot_pass_by_replaying_tf_directly(monkeypatch, tmp_path, consumer):
    """Controlled input/output defects are rejected even while direct TF remains correct."""
    calls = paper_adapters(monkeypatch, corrupt_faj=consumer == "faj", corrupt_matf=consumer == "matf")
    mapping = {"faj_q3": tmp_path / "faj", "matf_current": tmp_path / "matf"}
    expected = replay.labelled(replay.diagnose("tradablefactors"))
    with pytest.raises(AssertionError):
        actual = replay.consumer_paper_results(mapping, tmp_path)
        replay.check(replay.labelled(actual["faj_q3"]), expected, atol=1e-10, rtol=1e-8)
    assert calls[0][0] == "faj"


def test_paper_imports_restore_colliding_siblings(monkeypatch, tmp_path):
    """Different research trees cannot reuse each other's generic local_path module."""
    previous = ModuleType("local_path")
    previous.value = "pre-existing"
    monkeypatch.setitem(sys.modules, "local_path", previous)
    for name in ("first", "second"):
        directory = tmp_path / "papers" / name / "replication"
        directory.mkdir(parents=True)
        (directory / "local_path.py").write_text(f"value = {name!r}\n", encoding="utf-8")
        (directory / "runner.py").write_text("from local_path import value\n", encoding="utf-8")
        with replay.paper_module(tmp_path, name, "runner") as module:
            assert module.value == name
        assert sys.modules["local_path"] is previous


def test_operational_inventory_accounts_for_every_frozen_test_file():
    """Current and historical selections are explicit without editing the baseline identity."""
    directory = Path(__file__).parent
    inventory = replay.load_consumer_inventory(directory / "consumer_manifest.json", directory / "manifest.json")
    assert set(inventory["consumer_entry_points"]) == {"faj_q3", "matf_current", "rosaa"}


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["git-lf", "frozen-crlf"])
def test_operational_inventory_accepts_git_line_endings(tmp_path, newline):
    """A Git checkout preserves the frozen content while changing text line endings."""
    directory = Path(__file__).parent
    manifest = tmp_path / "manifest.json"
    original = (directory / "manifest.json").read_bytes().replace(b"\r\n", b"\n")
    manifest.write_bytes(original.replace(b"\n", newline))
    inventory = replay.load_consumer_inventory(directory / "consumer_manifest.json", manifest)
    assert inventory["reference_manifest_sha256"] == (
        "fd64f309c58002bca7e70ad2392d58ccc0be64c825d7c062a663f4f542930dda")


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["git-lf", "frozen-crlf"])
@pytest.mark.parametrize("changed", ["content", "frozen_digest", "lf_digest"])
def test_operational_inventory_rejects_changed_manifest_identity(tmp_path, newline, changed):
    """Line-ending portability cannot permit changed tolerances or unbound identity fields."""
    directory = Path(__file__).parent
    inventory = json.loads((directory / "consumer_manifest.json").read_text(encoding="utf-8-sig"))
    original = (directory / "manifest.json").read_bytes().replace(b"\r\n", b"\n")
    if changed == "content":
        reference = json.loads(original)
        reference["tolerances"]["atol"] *= 10
        original = json.dumps(reference, indent=2).encode("utf-8")
    elif changed == "frozen_digest":
        inventory["reference_manifest_sha256"] = "0" * 64
    else:
        inventory["reference_manifest_lf_sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(original.replace(b"\n", newline))
    path = tmp_path / "consumer_manifest.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable reference manifest"):
        replay.load_consumer_inventory(path, manifest)


def test_dropping_a_historical_contract_without_classification_is_rejected(tmp_path):
    """An unavailable old fixture cannot silently disappear from the migration inventory."""
    directory = Path(__file__).parent
    inventory = json.loads((directory / "consumer_manifest.json").read_text(encoding="utf-8-sig"))
    inventory["historical_tests"] = [item for item in inventory["historical_tests"]
                                      if not item["path"].endswith("test_reference_values.py")]
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="every original paper test file"):
        replay.load_consumer_inventory(path, directory / "manifest.json")
