"""Build the public FactorModels documentation with MyST."""

import os
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

project = "FactorModels"
release = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = release
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
extensions = ["myst_parser"]
html_theme = "furo"
html_title = f"factormodels {release} - factor-model construction, estimation and GLS analytics"
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]

# These two publishers return HTTP 403 to automated checks. Keep the references
# visible for manual verification, without excluding other publisher URLs.
linkcheck_ignore = [
    r"https://doi\.org/10\.2469/faj\.v51\.n3\.1904$",
    r"https://papers\.ssrn\.com/sol3/papers\.cfm\?abstract_id=6785958$",
]
linkcheck_retries = 3
linkcheck_timeout = 20

# Keep development, stable and tagged sources under their own canonical URLs.
html_baseurl = (
    os.environ.get("READTHEDOCS_CANONICAL_URL")
    or "https://factormodels.readthedocs.io/en/latest/"
)


class _CanonicalHead(HTMLParser):
    """Read canonical and indexing metadata from the HTML actually being published."""

    def __init__(self) -> None:
        super().__init__()
        self.canonicals = []
        self.noindex = False
        self.redirect = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href"))
        if tag == "meta" and attributes.get("name", "").lower() == "robots":
            self.noindex |= "noindex" in attributes.get("content", "").lower()
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            self.redirect = True


def _write_canonical_sitemap(app: Any, exception: Exception | None) -> None:
    """Offer only rendered, self-canonical pages from this build's own channel."""
    if exception is not None or app.builder.name != "html" or not app.config.html_baseurl:
        return
    base = app.config.html_baseurl.rstrip("/") + "/"
    urls = set()
    for docname in app.env.found_docs:
        head = _CanonicalHead()
        file = Path(app.builder.get_outfilename(docname))
        head.feed(file.read_text(encoding="utf-8").split("</head>")[0])
        if head.noindex or head.redirect or len(head.canonicals) != 1:
            continue
        uri = app.builder.get_target_uri(docname)
        expected = urljoin(base, quote("" if uri == "index.html" else uri, safe="/%"))
        if head.canonicals == [expected]:
            urls.add(expected)
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    root = ElementTree.Element(f"{{{namespace}}}urlset")
    for url in sorted(urls):
        entry = ElementTree.SubElement(root, f"{{{namespace}}}url")
        ElementTree.SubElement(entry, f"{{{namespace}}}loc").text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / "sitemap.xml", encoding="utf-8", xml_declaration=True,
        default_namespace=namespace,
    )


def _use_root_canonical(
    app: Any, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Use the version root as the homepage's canonical URL."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def _github_linkcheck_anchor(app: Any, uri: str) -> str | None:
    """Check GitHub heading IDs with their static user-content prefix."""
    parts = urlsplit(uri)
    if (
        parts.scheme == "https"
        and parts.netloc == "github.com"
        and parts.fragment
        and not parts.fragment.startswith("user-content-")
        and ("/blob/" in parts.path or len(parts.path.strip("/").split("/")) == 2)
    ):
        return urlunsplit(parts._replace(fragment="user-content-" + parts.fragment))
    return None


def setup(app: Any) -> None:
    """Register canonical page metadata and the matching sitemap producer."""
    app.connect("html-page-context", _use_root_canonical)
    app.connect("build-finished", _write_canonical_sitemap)
    app.connect("linkcheck-process-uri", _github_linkcheck_anchor)
