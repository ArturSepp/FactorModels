"""Installed metadata must enforce interpreter, security and licence floors."""

from importlib import metadata

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
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


def test_python_requirement_excludes_unsupported_interpreters():
    """Installers must reject Python 3.10 while admitting the supported minor lines."""
    distribution = metadata.metadata("factormodels")
    supported = SpecifierSet(distribution["Requires-Python"])
    assert not supported.contains("3.10.99")
    for interpreter in ("3.11.0", "3.12.0", "3.13.0", "3.14.0"):
        assert supported.contains(interpreter), interpreter
    classifiers = distribution.get_all("Classifier")
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers


def test_base_requirement_excludes_vulnerable_pillow_releases():
    """Every supported install must require the fixed imaging dependency."""
    requirements = [Requirement(item) for item in metadata.requires("factormodels")]
    pillow = [item for item in requirements if item.name == "pillow"]
    assert len(pillow) == 1
    requirement = pillow[0]
    assert requirement.marker is None
    assert not requirement.extras
    for vulnerable_release in ("11.3.0", "12.2.0", "12.3.0"):
        assert not requirement.specifier.contains(vulnerable_release), vulnerable_release
    assert requirement.specifier.contains("13.0.0")
    assert Version(metadata.version("pillow")) >= Version("13.0.0")
