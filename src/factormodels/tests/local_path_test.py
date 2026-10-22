"""Configured local resources and outputs work in source and installed packages."""
from pathlib import Path

import pytest
import yaml

import factormodels as fm


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Isolate settings and the working directory from the developer checkout."""
    from factormodels.utils import local_path
    config = tmp_path / "settings.yaml"
    config.write_text("RESOURCE_PATH: inputs\nOUTPUT_PATH: outputs\n", encoding="utf-8")
    monkeypatch.setenv("FACTORMODELS_SETTINGS_PATH", str(config))
    monkeypatch.delenv("FACTORMODELS_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("FACTORMODELS_RESOURCE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    local_path.get_paths.cache_clear()
    yield local_path, config
    local_path.get_paths.cache_clear()


def test_yaml_paths_are_relative_to_settings_not_working_directory(paths, tmp_path, monkeypatch):
    """Relative YAML roots are stable when an application changes its working directory."""
    local_path, config = paths
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert Path(local_path.get_resource_path()) == config.parent / "inputs"
    assert not (config.parent / "inputs").exists()
    assert Path(local_path.get_output_path()) == config.parent / "outputs"
    assert (config.parent / "outputs").is_dir()
    assert not (elsewhere / "outputs").exists()


def test_environment_overrides_yaml(paths, tmp_path, monkeypatch):
    """Explicit resource and output overrides take precedence over YAML."""
    local_path, _ = paths
    monkeypatch.setenv("FACTORMODELS_RESOURCE_PATH", str(tmp_path / "override-inputs"))
    monkeypatch.setenv("FACTORMODELS_OUTPUT_PATH", str(tmp_path / "override-outputs"))
    assert Path(local_path.get_resource_path()) == tmp_path / "override-inputs"
    assert Path(local_path.get_output_path()) == tmp_path / "override-outputs"
    assert not (tmp_path / "override-inputs").exists()


def test_relative_bundle_roundtrip_and_no_overwrite(paths, tmp_path):
    """Bundle names use YAML roots and refuse to overwrite an existing saved bundle."""
    local_path, config = paths
    config.write_text("RESOURCE_PATH: outputs\nOUTPUT_PATH: outputs\n", encoding="utf-8")
    local_path.get_paths.cache_clear()
    original = fm.synthetic_bundle()
    output = fm.save_bundle(original, "nested/synthetic")
    assert output == tmp_path / "outputs/nested/synthetic"
    loaded = fm.load_bundle("nested/synthetic")
    assert loaded.settings == original.settings
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        fm.save_bundle(original, "nested/synthetic")
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_absolute_bundle_paths_bypass_local_settings(paths, tmp_path):
    """An explicit absolute path retains its caller-selected location."""
    _, config = paths
    config.write_text("not: [valid", encoding="utf-8")
    output = fm.save_bundle(fm.synthetic_bundle(), tmp_path / "absolute/nested/bundle")
    assert fm.load_bundle(output).settings.recipe_id == fm.synthetic_bundle().settings.recipe_id
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("content", ["RESOURCE_PATH: null\nOUTPUT_PATH: null\n", "", "{}", "RESOURCE_PATH: '..'\nOUTPUT_PATH: '..'\n"])
def test_empty_settings_use_portable_defaults(paths, tmp_path, monkeypatch, content):
    """Installed defaults use cwd without requiring a writable site-packages directory."""
    local_path, config = paths
    config.write_text(content, encoding="utf-8")
    monkeypatch.setattr(local_path, "_checkout_root", lambda: None)
    assert Path(local_path.get_resource_path()) == tmp_path
    assert Path(local_path.get_output_path()) == tmp_path


def test_missing_explicit_settings_is_an_error(paths):
    """A misspelled explicit settings file must not silently select another output root."""
    local_path, config = paths
    config.unlink()
    with pytest.raises(FileNotFoundError):
        local_path.get_resource_path()


@pytest.mark.parametrize("content,error", [("- inputs\n- outputs\n", TypeError), ("RESOURCE_PATH: [inputs]\n", TypeError), ("not: [valid", yaml.YAMLError)])
def test_invalid_settings_fail_clearly(paths, content, error):
    """Malformed or wrongly shaped local path settings are never coerced to paths."""
    local_path, config = paths
    config.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        local_path.get_resource_path()


def test_output_creation_error_never_falls_back(paths, tmp_path):
    """An unwritable configured destination cannot redirect results elsewhere."""
    local_path, config = paths
    (tmp_path / "blocked").write_text("keep", encoding="utf-8")
    config.write_text("OUTPUT_PATH: blocked/child\n", encoding="utf-8")
    with pytest.raises(OSError):
        local_path.get_output_path()
    assert not (tmp_path / "outputs").exists()


def test_shipped_yaml_and_public_module_paths():
    """Wheel users receive the settings template and stable Python import paths."""
    import importlib
    package = Path(fm.__file__).parent
    assert (package / "settings.yaml").is_file()
    assert sorted(p.name for p in package.glob("*.py")) == ["__init__.py"]
    for name in ("bundle", "construction", "constraints", "estimation", "examples", "model", "portfolio_context", "risk", "signals", "spec"):
        module = importlib.import_module(f"factormodels.{name}")
        assert Path(module.__file__).name == "__init__.py"
    assert fm.FactorModelSpec.__module__ == "factormodels.spec"
    assert fm.FactorCovarEstimator.__module__ == "factormodels.estimation"


def test_output_resolution_has_no_creation_side_effect(paths, tmp_path):
    """Resolving a future output name does not create folders before caller validation."""
    local_path, _ = paths
    assert local_path.resolve_output_path("future/report.pdf") == tmp_path / "outputs/future/report.pdf"
    assert not (tmp_path / "outputs").exists()


def test_filesystem_root_is_an_explicit_resource_path(paths, tmp_path):
    """A filesystem root is an absolute setting, not an empty placeholder."""
    local_path, config = paths
    root = Path(tmp_path.anchor)
    config.write_text(yaml.safe_dump({"RESOURCE_PATH": root.as_posix()}), encoding="utf-8")
    assert Path(local_path.get_resource_path()) == root
