"""Resolve local resource and output paths using the OP YAML settings convention.

Paths in YAML are relative to the settings file; file names passed by callers
are relative to RESOURCE_PATH or OUTPUT_PATH. Absolute caller paths are retained.
FACTORMODELS_SETTINGS_PATH selects an external YAML file. Resource/output
environment overrides take precedence. Reading input paths never creates files.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

__all__ = ["get_paths", "get_resource_path", "get_output_path", "resolve_resource_path",
           "resolve_output_path", "create_output_directory"]

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_SETTINGS_PATH = _PACKAGE_DIR / "settings.yaml"


def _settings_path() -> Path:
    """Return the explicit local settings file or shipped template."""
    override = os.environ.get("FACTORMODELS_SETTINGS_PATH", "").strip()
    return Path(override).expanduser().resolve() if override else _SETTINGS_PATH


def _checkout_root() -> Path | None:
    """Locate the src-layout checkout without assuming the process working directory."""
    source = _PACKAGE_DIR.parent
    candidate = source.parent
    return candidate if source.name == "src" and (candidate / "pyproject.toml").is_file() else None


@lru_cache(maxsize=1)
def get_paths() -> dict:
    """Read YAML once; call cache_clear after changing settings in a running process."""
    settings = _settings_path()
    if not settings.is_file() and not os.environ.get("FACTORMODELS_SETTINGS_PATH", "").strip():
        return {}
    with settings.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("settings.yaml must contain a mapping")
    return values


def _configured_path(key: str) -> Path:
    """Resolve one override, YAML setting or checkout-aware default."""
    override = os.environ.get(f"FACTORMODELS_{key}", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    value = get_paths().get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a path string or null")
    normalized = "" if value is None else value.strip().replace(chr(92), "/")
    if normalized in {"", "..", "../"}:
        checkout = _checkout_root()
        return (checkout / "outputs" if checkout and key == "OUTPUT_PATH" else checkout or Path.cwd()).resolve()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _settings_path().parent / path).resolve()


def get_resource_path() -> str:
    """Return the resource root as an absolute portable path without creating it."""
    return _configured_path("RESOURCE_PATH").as_posix()


def get_output_path() -> str:
    """Create and return the configured output root, propagating filesystem errors."""
    path = _configured_path("OUTPUT_PATH")
    path.mkdir(parents=True, exist_ok=True)
    return path.as_posix()


def resolve_resource_path(path: str | Path) -> Path:
    """Resolve a supplied input name below the resource root, retaining absolute paths."""
    target = Path(path).expanduser()
    return (target if target.is_absolute() else Path(get_resource_path()) / target).resolve()


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a supplied output name below the output root, retaining absolute paths."""
    target = Path(path).expanduser()
    return (target if target.is_absolute() else _configured_path("OUTPUT_PATH") / target).resolve()


def create_output_directory(directory: str | Path) -> Path:
    """Create a new configured output directory and parents, refusing existing output."""
    target = resolve_output_path(directory)
    target.mkdir(parents=True, exist_ok=False)
    return target
