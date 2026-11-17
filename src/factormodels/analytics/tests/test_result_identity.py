"""Identity contracts for public containers holding pandas objects."""
from copy import deepcopy

import pytest

import factormodels as fm
from factormodels.analytics.tests.test_model_accessors import _snapshot
from factormodels.estimation.tests.test_equity_fx import augment
from factormodels.estimation.tests.test_equity_payoff import assemble, fixture


@pytest.mark.parametrize('kind', [
    'snapshot', 'rolling', 'diagnostics', 'dated_covariance', 'equity_fx', 'equity_payoff',
])
def test_result_identity_does_not_compare_hash_or_print_pandas_tables(kind):
    """Equal-valued tables remain distinct, usable dictionary keys with compact displays."""
    snapshot = _snapshot()
    if kind == 'rolling':
        result = fm.RollingFactorModel({snapshot.cutoff: snapshot})
    elif kind == 'diagnostics':
        result = fm.PreparedSnapshotDiagnostics(
            snapshot.fit_statistics, cutoff=snapshot.cutoff, alpha_settings={},
            alpha_source='unavailable', alpha_available=False,
        )
    elif kind == 'dated_covariance':
        result = fm.DatedFactorCovariance(
            snapshot.factor_covariance, snapshot.cutoff, snapshot.cutoff,
            snapshot.factor_return_types,
        )
    elif kind == 'equity_fx':
        result = augment()
    elif kind == 'equity_payoff':
        parts, _ = fixture()
        result = assemble(parts)
    else:
        result = snapshot
    other = deepcopy(result)
    assert result == result
    assert result != other
    lookup = {result: 'original', other: 'copy'}
    assert lookup[result] == 'original' and len(lookup) == 2
    assert '\n' not in repr(result) and len(repr(result)) < 200
    assert type(result).__name__ in repr(result)
