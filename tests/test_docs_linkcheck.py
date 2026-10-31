"""Documentation link checks retain anchor validation and narrow exclusions."""

from pathlib import Path
import re
import runpy

import pytest


CONFIG = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docs/conf.py"))


def test_sphinx_registers_the_github_anchor_handler():
    """The real documentation setup must enable the checker adaptation."""
    class App:
        def __init__(self):
            self.handlers = {}

        def connect(self, event, callback):
            self.handlers[event] = callback

    app = App()
    CONFIG["setup"](app)
    assert app.handlers["linkcheck-process-uri"] is CONFIG["_github_linkcheck_anchor"]


@pytest.mark.parametrize("uri", [
    "https://github.com/ArturSepp/FactorModels#five-minute-quickstart",
    "https://github.com/ArturSepp/OptimalPortfolios/blob/main/docs/conventions.md#notation",
    "https://github.com/ArturSepp/FactorModels/blob/reviewed/candidate/README.md#missing-heading",
])
def test_github_heading_checks_use_static_ids_without_dropping_fragments(uri):
    """Existing and missing heading fragments both remain subject to checking."""
    address, fragment = uri.split("#")
    assert CONFIG["_github_linkcheck_anchor"](None, uri) == address + "#user-content-" + fragment


@pytest.mark.parametrize("uri", [
    "https://github.com/ArturSepp/FactorModels/blob/main/README.md#user-content-five-minute-quickstart",
    "https://github.com/ArturSepp/FactorModels/blob/main/README.md",
    "https://github.com/ArturSepp/FactorModels/issues/1#issuecomment-123",
    "https://github.com/ArturSepp/FactorModels/commit/" + "1" * 40 + "#diff-123",
    "https://www.sphinx-doc.org/en/master/usage/configuration.html#linkcheck-anchors",
    "https://github.com.example.org/ArturSepp/FactorModels#heading",
])
def test_unrelated_links_and_explicit_static_ids_are_unchanged(uri):
    """Only ordinary GitHub repository and blob heading fragments are adapted."""
    assert CONFIG["_github_linkcheck_anchor"](None, uri) is None


def test_only_the_two_observed_publisher_blocks_are_excluded():
    """New references, missing GitHub anchors and other publishers still get checked."""
    def ignored(uri):
        return any(re.match(pattern, uri) for pattern in CONFIG["linkcheck_ignore"])

    assert ignored("https://doi.org/10.2469/faj.v51.n3.1904")
    assert ignored("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958")
    for uri in (
        "https://doi.org/10.2469/faj.v51.n3.19040",
        "https://doi.org/10.1234/other",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=67859580",
        "https://github.com/ArturSepp/FactorModels#missing-heading",
        "https://www.gnu.org/licenses/gcc-exception-3.1.html",
    ):
        assert not ignored(uri)
