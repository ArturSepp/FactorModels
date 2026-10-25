"""Production ownership and finite debt; synthetic defects must be rejected."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fm_architecture_guard", ROOT / "tools/check_architecture.py")
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)
DEBT = json.loads((ROOT / "tools/architecture_debt.json").read_text(encoding="utf-8"))


def fixtures(**sources):
    return {"factormodels." + name: ("src/factormodels/" + name.replace(".", "/") + ".py", ast.parse(text))
            for name, text in sources.items()}


def violations(sources):
    return guard.check_imports(sources, check_deferred_inventory=False)[0]


def test_production_architecture_and_debt():
    """Every import and unsuppressed annotation/header has an exact reviewed owner."""
    assert guard.check()["errors"] == []
    assert DEBT["annotations"] == []
    assert DEBT["headers"] == {}


@pytest.mark.parametrize("statement", [
    "from factorlasso import LassoModel",
    "def f():\n    from factorlasso import CurrentFactorCovarData",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import factorlasso",
])
def test_direct_nested_and_type_checking_fl_imports_fail(statement):
    assert any("forbidden external" in e for e in violations(fixtures(**{"analytics.bad": statement})))


@pytest.mark.parametrize("statement", [
    "from ..estimation.fit import fit",
    "from ..estimation import fit",
    "import factormodels.estimation.fit",
])
def test_relative_and_absolute_reverse_dependencies_fail(statement):
    sources = fixtures(**{"analytics.bad": statement, "estimation.fit": "def fit(): pass"})
    assert any("forbidden internal" in e for e in violations(sources))


@pytest.mark.parametrize("statement", ["from factormodels import FactorCovarEstimator", "import factormodels as fm"])
def test_root_facade_bypass_fails(statement):
    assert any("root-facade bypass" in e for e in violations(fixtures(**{"analytics.bad": statement})))


def test_subpackage_reexport_cannot_hide_reverse_dependency():
    sources = fixtures(**{"analytics.bad": "from ..analytics import fit", "estimation.fit": "def fit(): pass"})
    sources["factormodels.analytics"] = (
        "src/factormodels/analytics/__init__.py", ast.parse("from ..estimation.fit import fit"))
    assert any("analytics.bad: forbidden internal" in e for e in violations(sources))


def test_relative_sibling_and_qis_risk_imports_are_allowed():
    sources = fixtures(**{"analytics.models": "from .risk import build\nimport qis",
                          "analytics.risk": "def build(): pass"})
    assert violations(sources) == []


def test_initializer_cycle_is_rejected_but_typing_edges_are_not_eager():
    sources = fixtures(**{"analytics.a": "from .b import b", "analytics.b": "from .a import a"})
    assert any("module-initialization cycles" in e for e in violations(sources))
    sources["factormodels.analytics.a"] = (
        "src/factormodels/analytics/a.py", ast.parse("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .b import b"))
    assert not any("module-initialization cycles" in e for e in violations(sources))


def test_unreviewed_deferred_cycle_fails():
    sources = fixtures(**{"analytics.a": "def a():\n    from .b import b", "analytics.b": "from .a import a"})
    errors, receipt = guard.check_imports(sources)
    assert receipt["eager_cycles"] == []
    assert any("unreviewed or stale deferred cycle" in e for e in errors)


def test_negated_typing_guard_keeps_runtime_cycle_visible():
    sources = fixtures(**{"analytics.a": "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    from .b import b",
                          "analytics.b": "from .a import a"})
    assert any("module-initialization cycles" in e for e in violations(sources))


def test_unlisted_legacy_module_is_not_a_new_permission():
    assert any("unknown production owner" in e for e in violations(fixtures(**{"premia.new": "pass"})))


@pytest.mark.parametrize("module", list(guard.CALIBRATION))
def test_named_calibration_exceptions_never_allow_fl_containers(module):
    name = module.removeprefix("factormodels.")
    assert violations(fixtures(**{name: "from factorlasso import gaussian_quadratic_summary"})) == []
    assert any("forbidden external" in e for e in violations(fixtures(**{name: "from factorlasso import CurrentFactorCovarData"})))


def test_final_boundary_has_no_gls_transition_permissions():
    assert not (ROOT / "tools/architecture_transition.json").exists()
    sources = fixtures(**{"analytics.opportunity": "from factorlasso import gaussian_quadratic_summary"})
    assert any("forbidden external" in e for e in violations(sources))
    sources = fixtures(**{"analytics.example": "from ..gls_diagnostic.geometry import compute_gls_geometry",
                          "gls_diagnostic.geometry": "pass"})
    assert any("forbidden internal" in e for e in violations(sources))


def test_ruff_catches_all_six_annotation_rules_including_variadics(tmp_path):
    source = tmp_path / "src/factormodels/analytics/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("class Example:\n    def __init__(self): pass\n"
                      "def public(value, *args, **kwargs): pass\ndef _private(): pass\n", encoding="utf-8")
    actual = guard.annotation_inventory(tmp_path)
    assert {row[2] for row in actual} == set(guard.ANN)
    assert guard.annotation_errors(actual, [])
    assert guard.annotation_errors(actual, actual) == []
    assert guard.annotation_errors(actual + [actual[0]], actual)  # same-file additions cannot hide
    assert guard.annotation_errors([], actual)  # remove stale exceptions when fixing debt


GOOD_HEADER = '''Purpose: Calculate a labelled conditional risk statistic.
Inputs: Supplied loadings and annual factor covariance.
Outputs: Labelled covariance and explicit timing metadata.
Units: Annual covariance in native factor return coordinates.
Conditioning: Supplied loadings are held fixed throughout.
Ownership: FM analytics owns assembly; QIS owns risk interchange.
'''


def test_header_requires_content_and_catches_a_removed_field():
    assert guard.header_deficit(ast.parse(repr(GOOD_HEADER)))["missing"] == []
    for field in guard.HEADER_FIELDS:
        damaged = "\n".join(line for line in GOOD_HEADER.splitlines() if not line.startswith(field + ":"))
        assert field in guard.header_deficit(ast.parse(repr(damaged)))["missing"]
    assert guard.header_deficit(ast.parse(repr("This is a long header.\n" * 20)))["missing"]
    assert "Units" in guard.header_deficit(ast.parse(repr(GOOD_HEADER.replace(
        "Units: Annual covariance in native factor return coordinates.", "Units: TODO"))))["missing"]


def test_new_module_missing_header_and_changed_existing_header_fail():
    sources = fixtures(**{"analytics.new": "def calculate(): pass"})
    assert guard.check_headers(sources, {})[0]
    sources = guard.read_sources(ROOT)
    relative, tree = sources["factormodels.analytics.risk"]
    tree.body[0].value.value = tree.body[0].value.value.replace("Units:", "Unit description:")
    assert any(relative in e for e in guard.check_headers(sources, DEBT["headers"])[0])


def test_ruff_exceptions_match_exact_production_debt_and_keep_other_checks():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]
    assert set(guard.ANN) <= set(config["select"])
    expected = {}
    for path, _, rule, _ in DEBT["annotations"]:
        expected.setdefault(path, set()).add(rule)
    actual = {path: set(rules) for path, rules in config["per-file-ignores"].items() if path.startswith("src/")}
    assert actual == expected
    for pattern in ("**/tests/**", "examples/**", "tools/**"):
        assert set(config["per-file-ignores"][pattern]) == set(guard.ANN)
    assert DEBT["remove_in_stage"] == 4


def test_variadic_inventory_is_current():
    actual = []
    for _, (relative, tree) in guard.read_sources(ROOT).items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for kind, arg in (("*args", node.args.vararg), ("**kwargs", node.args.kwarg)):
                    if arg is not None:
                        actual.append({"path": relative, "function": node.name, "parameter": arg.arg,
                                       "kind": kind, "annotated": arg.annotation is not None})
    assert actual == DEBT["variadic_inventory"]
