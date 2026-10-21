"""The installed package must exclude historical GPL FactorLasso releases."""

from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import Version


def test_base_requirement_excludes_pre_mit_factorlasso():
    """Check built metadata, including wheel installations outside the source tree."""
    requirements = [Requirement(item) for item in metadata.requires("factormodels")]
    factorlasso = [item for item in requirements if item.name == "factorlasso"]
    assert len(factorlasso) == 1
    requirement = factorlasso[0]
    assert requirement.marker is None
    assert not requirement.extras
    assert requirement.specifier.contains("1.0.4.post1")
    for old_release in ("0.1.0", "1.0.0", "1.0.3", "1.0.4"):
        assert not requirement.specifier.contains(old_release), old_release


def test_installed_factorlasso_uses_the_verified_mit_floor():
    """The resolved estimator has the required version and MIT distribution metadata."""
    distribution = metadata.distribution("factorlasso")
    assert Version(distribution.version) >= Version("1.0.4.post1")
    assert distribution.metadata["License-Expression"] == "MIT"
