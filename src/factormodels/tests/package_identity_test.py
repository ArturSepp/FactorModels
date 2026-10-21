"""Installed identity contracts for the clean FactorModels namespace migration."""
from importlib import metadata, util

import factormodels


def test_canonical_distribution_has_no_legacy_forwarding_package():
    """A clean installation exposes FactorModels without the removed namespace."""
    assert metadata.distribution("factormodels").metadata["Name"] == "factormodels"
    assert util.find_spec("tradablefactors") is None
    assert factormodels.FactorModelSpec.__name__ == "FactorModelSpec"
    assert factormodels.FactorBundle.__name__ == "FactorBundle"
    assert factormodels.FactorModelSnapshot.__name__ == "FactorModelSnapshot"
    assert factormodels.RollingFactorModel.__name__ == "RollingFactorModel"
