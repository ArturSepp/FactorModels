"""Parameter documentation for the completed-model workflow entry points."""
import inspect
import re
from enum import Enum

import pytest

import factormodels as fm


ENTRY_POINTS = [
    fm.FactorCovarEstimator.fit_current, fm.FactorCovarEstimator.fit_rolling,
    fm.FactorModelSnapshot, fm.RollingFactorModel, fm.snapshot_from_moments,
    fm.snapshots_from_factorlasso, fm.prepare_snapshot_diagnostics,
    fm.estimate_historical_alpha, fm.compute_cma_diagnostics, fm.FactorOpportunityModel,
]

# Enumeration constructors use Enum's framework signature, not package workflow arguments.
ROOT_CALLABLES = [
    getattr(fm, name) for name in fm.__all__
    if callable(getattr(fm, name))
    and getattr(getattr(fm, name), '__module__', '').startswith('factormodels')
    and not (inspect.isclass(getattr(fm, name)) and issubclass(getattr(fm, name), Enum))
]
ENTRY_POINTS = list(dict.fromkeys([*ENTRY_POINTS, *ROOT_CALLABLES]))


def _argument_names(docstring):
    """Read Google argument names, including optional parenthesized type annotations."""
    lines = inspect.cleandoc(docstring).splitlines()
    collecting = False
    names = set()
    for line in lines:
        if line == 'Args:':
            collecting = True
            continue
        if collecting and line and not line[0].isspace():
            collecting = False
        if collecting:
            match = re.match(r'^    (\w+)(?:\s+\(.+\))?:', line)
            if match:
                names.add(match.group(1))
    return names


@pytest.mark.parametrize('entry', ENTRY_POINTS, ids=lambda entry: entry.__qualname__)
def test_workflow_docstrings_describe_every_public_parameter(entry):
    """Signature parameters cannot disappear behind an Args heading with missing content."""
    expected = set(inspect.signature(entry).parameters) - {'self', 'cls'}
    assert expected == _argument_names(inspect.getdoc(entry)), entry.__qualname__


def test_argument_reader_accepts_typed_parameters_without_counting_return_fields():
    """Valid Google typed arguments are supported and return fields are separate."""
    doc = 'Summary.\n\nArgs:\n    value (pd.Series): Input.\n\nReturns:\n    output: Result.'
    assert _argument_names(doc) == {'value'}
    assert _argument_names(doc.replace('    value (pd.Series): Input.', '')) == set()
    assert _argument_names(doc.replace('pd.Series', 'array-like of shape (n, k)')) == {'value'}
