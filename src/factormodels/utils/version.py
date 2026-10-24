"""Installed distribution version, with an uninstalled-source fallback."""
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("factormodels")
except _PackageNotFoundError:
    __version__ = "0+unknown"
