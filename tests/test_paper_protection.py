"""Private paper workspaces must stay outside the public Git index."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_only_public_paper_guides_are_tracked():
    """Check NUL-delimited index paths in normal checkouts and linked worktrees."""
    root = Path(__file__).resolve().parents[1]
    if not (root / '.git').exists() or shutil.which('git') is None:
        pytest.skip('source archive without Git metadata; artifact exclusions are checked at build time')
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z', '--', 'papers'],
        check=True, capture_output=True,
    )
    tracked = {path.decode('utf-8') for path in result.stdout.split(b'\0') if path}
    assert tracked <= {'papers/README.md', 'papers/AGENTS.md'}, sorted(tracked)
