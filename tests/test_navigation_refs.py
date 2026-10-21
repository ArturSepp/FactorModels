"""Advertised navigation must follow its reviewed source, preserving provenance."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "navigation_refs", Path(__file__).resolve().parents[1] / "tools/check_navigation_refs.py"
)
navigation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(navigation)
BASE = "https://github.com/ArturSepp/FactorModels"
REF = "reviewed/candidate"


@pytest.fixture
def source(tmp_path):
    """Create source navigation with a real local target and independent history."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        f"[Guide]({BASE}/blob/{REF}/docs/index.md)\n"
        f"First recorded: [date]({BASE}/commit/" + "1" * 40 + ")\n"
        "[Other source](https://github.com/Other/Repo/blob/old/README.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        f'[project.urls]\nDocumentation = "{BASE}/tree/{REF}/docs"\n', encoding="utf-8"
    )
    (tmp_path / "tools/navigation_refs.json").write_text(json.dumps({
        "repository": BASE, "reviewed_ref": REF, "evidence_links": [],
    }), encoding="utf-8")
    return tmp_path


def test_historical_commits_other_repositories_and_branch_slashes_are_allowed(source):
    """Preparation validates navigation without claiming current release identity."""
    result = navigation.check_navigation(source)
    assert result == {"reviewed_ref": REF, "checked_links": 2, "release_source_verified": False}


@pytest.mark.parametrize("document", ["README.md", "pyproject.toml", "docs/index.md"])
def test_obsolete_advertised_ref_is_rejected_on_each_surface(source, document):
    """An old source ref must fail even when its target filename still exists."""
    path = source / document
    path.write_text(path.read_text(encoding="utf-8") + f"\n{BASE}/blob/obsolete/docs/index.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="obsolete navigation ref"):
        navigation.check_navigation(source)


def test_exact_explained_immutable_evidence_is_allowed(source):
    """A legitimate immutable blob is exempted individually with a review reason."""
    url = BASE + "/blob/" + "2" * 40 + "/evidence.json"
    path = source / "tools/navigation_refs.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["evidence_links"] = [{"url": url, "reason": "Published reference evidence"}]
    path.write_text(json.dumps(policy), encoding="utf-8")
    (source / "docs/index.md").write_text(f"[Evidence]({url})\n", encoding="utf-8")
    assert navigation.check_navigation(source)["checked_links"] == 2
    (source / "docs/index.md").write_text(f"[Stale guide]({url}x)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="obsolete navigation ref"):
        navigation.check_navigation(source)


def test_missing_target_and_empty_navigation_are_rejected(source):
    """Navigation cannot pass solely because it points at a nonexistent file."""
    (source / "docs/index.md").unlink()
    with pytest.raises(ValueError, match="missing advertised source target"):
        navigation.check_navigation(source)
    (source / "README.md").write_text("# No links\n", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project.urls]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No advertised source navigation"):
        navigation.check_navigation(source)


def test_release_requires_same_commit_and_clean_tracked_source(source, monkeypatch):
    """Clean current source passes; stale identities and dirty files fail closed."""
    responses = iter(["current", "current", ""])
    monkeypatch.setattr(navigation, "_git", lambda *args: next(responses))
    assert navigation.check_navigation(source, require_current_source=True)["release_source_verified"]
    responses = iter(["current", "old"])
    with pytest.raises(ValueError, match="does not identify the release source"):
        navigation.check_navigation(source, require_current_source=True)
    responses = iter(["current", "current", " M README.md"])
    with pytest.raises(ValueError, match="Tracked release source is dirty"):
        navigation.check_navigation(source, require_current_source=True)
