"""Check advertised source navigation without rewriting immutable historical evidence.

Preparation checks use the explicitly reviewed reference in navigation_refs.json.
Release checks additionally require that reference to identify the clean checkout.
No network access or Git mutation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


URL = re.compile(r"https://github\.com/[^\s<>\[\](){}\"'`]+")


def _git(root: Path, *args: str) -> str:
    """Read a Git identity or tracked working-tree state without changing it."""
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", timeout=30,
    ).stdout.strip()


def check_navigation(root: Path, *, require_current_source: bool = False) -> dict:
    """Reject stale advertised refs, allowing exact explained evidence exceptions."""
    root = root.resolve()
    policy = json.loads((root / "tools/navigation_refs.json").read_text(encoding="utf-8"))
    repository = urlsplit(policy["repository"])
    expected = policy["reviewed_ref"]
    if (repository.scheme != "https" or repository.netloc != "github.com"
            or len(repository.path.strip("/").split("/")) != 2
            or not isinstance(expected, str) or not expected
            or any(character.isspace() for character in expected)):
        raise ValueError("An explicit GitHub repository and reviewed reference are required")
    base = repository.path.rstrip("/") + "/"
    evidence = {}
    for item in policy.get("evidence_links", []):
        if not item.get("reason", "").strip() or not item.get("url"):
            raise ValueError("Each immutable evidence exception needs an exact URL and reason")
        evidence[item["url"]] = item["reason"]
    documents = [root / "README.md", root / "pyproject.toml", *sorted((root / "docs").rglob("*.md"))]
    if not (root / "docs").is_dir():
        raise ValueError("Missing advertised docs directory")
    checked = []
    errors = []
    for document in documents:
        text = document.read_text(encoding="utf-8-sig")
        # Only project URL metadata is navigation; other TOML sections may record evidence.
        if document.name == "pyproject.toml":
            section = re.search(r"(?ms)^\[project\.urls\]\s*\n(.*?)(?=^\[|\Z)", text)
            if section is None:
                raise ValueError("Missing project URL metadata")
            text = section[1]
        for match in URL.finditer(text):
            url = match[0]
            parsed = urlsplit(url)
            path = unquote(parsed.path)
            if parsed.netloc != repository.netloc or not path.startswith(base):
                continue
            tail = path[len(base):]
            if not tail.startswith(("blob/", "tree/")):
                continue  # /commit/ First recorded and other provenance links are retained.
            if url in evidence:
                continue
            kind, reference_path = tail.split("/", 1)
            label = str(document.relative_to(root))
            if not reference_path.startswith(expected + "/"):
                errors.append(f"{label}: obsolete navigation ref; expected {expected}: {url}")
                continue
            target = reference_path[len(expected) + 1:]
            local = root / target
            if not local.resolve().is_relative_to(root) or not local.exists():
                errors.append(f"{label}: missing advertised source target: {url}")
            elif (kind == "blob") != local.is_file():
                errors.append(f"{label}: source navigation kind does not match target: {url}")
            checked.append({"document": label, "url": url})
    if not checked:
        errors.append("No advertised source navigation was checked")
    if errors:
        raise ValueError("\n".join(errors))
    if require_current_source:
        current = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
        reviewed = _git(root, "rev-parse", "--verify", "--end-of-options", expected + "^{commit}")
        if reviewed != current:
            raise ValueError("Advertised navigation reference does not identify the release source")
        if _git(root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Tracked release source is dirty")
    return {"reviewed_ref": expected, "checked_links": len(checked),
            "release_source_verified": require_current_source}


def main() -> None:
    """Validate local preparation or enforce exact-source navigation during release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-current-source", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_navigation(args.root, require_current_source=args.require_current_source)))


if __name__ == "__main__":
    main()
