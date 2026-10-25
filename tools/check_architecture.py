"""Check production import ownership, deferred cycles and scoped refactor debt.

The AST walk includes function-local and type-checking imports. Standard-library
imports are allowed independently of the third-party and FM ownership matrices.
Package re-exports are resolved before checking ownership, so a facade cannot
conceal a reverse dependency. Only module-initialization edges enter the eager
cycle graph; separately enumerated deferred cycles remain visible in the receipt.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ANN = ("ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204")
INTERNAL = {
    "analytics": {"analytics", "utils"},
    "estimation": {"estimation", "analytics", "utils"},
    "gls_diagnostic": {"gls_diagnostic", "analytics", "utils"},
    "reports": {"reports", "gls_diagnostic", "analytics", "utils"},
    "utils": {"utils"},
}
EXTERNAL = {
    "analytics": {"numpy", "pandas", "qis"},
    "estimation": {"numpy", "pandas", "qis", "factorlasso"},
    "gls_diagnostic": {"numpy", "pandas", "scipy", "qis"},
    "reports": {"numpy", "pandas", "qis", "matplotlib"},
    "utils": {"yaml"},
}
CALIBRATION = {
    "factormodels.gls_diagnostic.opportunity": {"gaussian_quadratic_summary"},
    "factormodels.gls_diagnostic.universe_diagnostics": {"gaussian_quadratic_summary"},
    "factormodels.gls_diagnostic.alpha_uncertainty": {
        "gaussian_quadratic_summary", "sample_gaussian_estimates",
        "linear_confidence_intervals", "quadratic_confidence_summary",
    },
}
HEADER_FIELDS = ("Purpose", "Inputs", "Outputs", "Units", "Conditioning", "Ownership")
# Export-only initializers and test packages are excluded structurally. These
# two non-numerical infrastructure modules have descriptive ordinary docstrings.
HEADER_EXEMPT = {
    "src/factormodels/utils/local_path.py": "Filesystem/settings resolution; no numerical analytics.",
    "src/factormodels/utils/version.py": "Distribution metadata lookup and source fallback.",
}


@dataclass(frozen=True)
class Access:
    """One import alias, with its execution context retained."""
    source: str
    module: str
    symbol: str
    context: str

    def key(self):
        return [self.source, self.module, self.symbol, self.context]


def read_sources(root):
    """Read production modules only, without importing the package."""
    result = {}
    for path in sorted((root / "src/factormodels").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if "tests" in path.relative_to(root / "src/factormodels").parts:
            continue
        module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        result[module] = (relative, ast.parse(path.read_text(encoding="utf-8-sig")))
    return result


def imports(module, relative, tree):
    """Resolve all relative imports, including nested and TYPE_CHECKING uses."""
    package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
    found = []

    def visit(node, context="eager"):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Defaults/decorators can execute eagerly; imports occur in the body.
            context = "deferred" if context != "typing" else context
        if isinstance(node, ast.Import):
            found.extend(Access(module, n.name, "", context) for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, package)
            found.extend(Access(module, name, n.name, context) for n in node.names)
        condition = node.test if isinstance(node, ast.If) else None
        negated = isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not)
        atom = condition.operand if negated else condition
        type_guard = (isinstance(atom, ast.Name) and atom.id == "TYPE_CHECKING"
                      or isinstance(atom, ast.Attribute) and atom.attr == "TYPE_CHECKING")
        if type_guard:
            # Compound conditions remain conservative eager edges. Handle the
            # ordinary guard and its negation without hiding the runtime branch.
            for child in node.body:
                visit(child, context if negated else "typing")
            for child in node.orelse:
                visit(child, "typing" if negated else context)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, context)

    visit(tree)
    return found


def export_bindings(sources):
    """Map initializer re-exports to the actual defining module/symbol."""
    bindings = {}
    for module, (relative, tree) in sources.items():
        if not relative.endswith("/__init__.py"):
            continue
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                accesses = imports(module, relative, ast.Module(body=[node], type_ignores=[]))
                for alias, access in zip(node.names, accesses):
                    bindings[module, alias.asname or alias.name] = (access.module, access.symbol)
    return bindings


def resolved_target(access, sources, bindings):
    module, symbol = access.module, access.symbol
    visited = set()
    while (module, symbol) in bindings and (module, symbol) not in visited:
        visited.add((module, symbol))
        module, symbol = bindings[module, symbol]
    if symbol and module + "." + symbol in sources:
        return module + "." + symbol
    return module


def cycles(graph):
    """Return strongly connected components, excluding trivial acyclic nodes."""
    index, low, stack, active, result = {}, {}, [], set(), []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        active.add(node)
        for target in sorted(graph.get(node, ())):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in graph.get(node, ()):
                result.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return sorted(result)


def check_imports(sources, *, check_deferred_inventory=True):
    """Enforce the final physical ownership matrix without transition exceptions."""
    bindings = export_bindings(sources)
    eager, all_edges, errors = {}, {}, []

    def owner(module):
        return module.split(".")[1] if module.startswith("factormodels.") else ""

    for source, (relative, tree) in sources.items():
        layer = owner(source)
        if source != "factormodels" and layer not in INTERNAL:
            errors.append(f"{source}: unknown production owner")
        for access in imports(source, relative, tree):
            target = resolved_target(access, sources, bindings)
            top = access.module.split(".")[0]
            if top == "factormodels":
                if source != "factormodels" and access.module == "factormodels":
                    errors.append(f"{source}: root-facade bypass {access.key()}")
                if access.symbol == "*":
                    errors.append(f"{source}: wildcard import obscures ownership")
                if target not in sources:
                    errors.append(f"{source}: unresolved internal module {target}")
                # Only the root facade exports across all layers.
                elif source != "factormodels" and owner(target) not in INTERNAL.get(layer, set()):
                    errors.append(f"{source}: forbidden internal dependency {target}")
                # Keep the facade node in the cycle graph; it executes its own
                # imports. Named child-module imports point directly to that child.
                edge = access.module
                if access.symbol and edge + "." + access.symbol in sources:
                    edge += "." + access.symbol
                if edge in sources:
                    all_edges.setdefault(source, set()).add(edge)
                    if access.context == "eager":
                        eager.setdefault(source, set()).add(edge)
            elif top not in sys.stdlib_module_names:
                allowed = top in EXTERNAL.get(layer, set())
                if top == "factorlasso" and layer != "estimation":
                    allowed = access.module == "factorlasso" and access.symbol in CALIBRATION.get(source, set())
                if not allowed:
                    errors.append(f"{source}: forbidden external import {access.module}:{access.symbol}")
    eager_cycles = cycles(eager)
    if eager_cycles:
        errors.append(f"module-initialization cycles: {eager_cycles}")
    deferred = cycles(all_edges)
    expected = [[
        "factormodels.analytics.models", "factormodels.analytics.risk",
    ]]
    # The permanent model/risk pair is a local-method convenience import, not an
    # initialization cycle. It must disappear from this list if removed in code.
    if check_deferred_inventory and deferred != expected:
        errors.append(f"unreviewed or stale deferred cycle inventory: {deferred}")
    return errors, {"eager_cycles": eager_cycles, "deferred_cycles": deferred}


def annotation_inventory(root):
    """Run the six Ruff rules without per-file/noqa suppression and bind deficits to symbols."""
    command = [sys.executable, "-m", "ruff", "check", "src/factormodels", "--isolated",
               "--ignore-noqa", "--select", ",".join(ANN), "--output-format", "json"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    scopes = {}
    for module, (relative, tree) in read_sources(root).items():
        locations = []

        def walk(node, scope=()):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope += (node.name,)
                locations.append((node.lineno, node.end_lineno, ".".join(scope)))
            for child in ast.iter_child_nodes(node):
                walk(child, scope)

        walk(tree)
        scopes[relative] = locations
    inventory = []
    for diagnostic in json.loads(result.stdout):
        relative = Path(diagnostic["filename"]).relative_to(root).as_posix()
        if relative not in scopes:
            continue
        row = diagnostic["location"]["row"]
        containing = [(end - start, name) for start, end, name in scopes[relative] if start <= row <= end]
        symbol = min(containing)[1] if containing else "<module>"
        inventory.append([relative, symbol, diagnostic["code"], diagnostic["message"]])
    return sorted(inventory)


def annotation_errors(actual, expected):
    observed, permitted = Counter(map(tuple, actual)), Counter(map(tuple, expected))
    return ([f"new annotation debt: {row}" for row in (observed - permitted).elements()]
            + [f"stale annotation exception: {row}" for row in (permitted - observed).elements()])


def header_deficit(tree):
    """Require six substantive labelled fields, not a minimum docstring length."""
    doc = ast.get_docstring(tree) or ""
    fields = dict(re.findall(
        r"^(Purpose|Inputs|Outputs|Units|Conditioning|Ownership):\s*(.*?)(?=^[A-Z][A-Za-z /]+:|\Z)",
        doc, flags=re.MULTILINE | re.DOTALL,
    ))
    missing = [name for name in HEADER_FIELDS if len(fields.get(name, "").split()) < 3
               or re.search(r"\b(TODO|TBD|placeholder)\b", fields.get(name, ""), re.IGNORECASE)]
    return {"missing": missing, "doc_sha256": hashlib.sha256(doc.encode()).hexdigest()}


def check_headers(sources, debt):
    errors, actual = [], {}
    for _, (relative, tree) in sources.items():
        if relative.endswith("/__init__.py") or relative in HEADER_EXEMPT:
            continue
        deficit = header_deficit(tree)
        if deficit["missing"]:
            actual[relative] = deficit
    for relative in sorted(actual.keys() | debt.keys()):
        if actual.get(relative) != debt.get(relative):
            errors.append(f"{relative}: new/changed numerical header deficit or stale exception")
    return errors, actual


def check(root=ROOT):
    debt = json.loads((root / "tools/architecture_debt.json").read_text(encoding="utf-8"))
    sources = read_sources(root)
    errors, receipt = check_imports(sources)
    annotations = annotation_inventory(root)
    errors += annotation_errors(annotations, debt["annotations"])
    header_errors, headers = check_headers(sources, debt["headers"])
    errors += header_errors
    receipt.update(annotation_deficits=len(annotations), header_deficits=len(headers), errors=errors)
    return receipt


if __name__ == "__main__":
    receipt = check()
    print(json.dumps(receipt, indent=2))
    raise SystemExit(bool(receipt["errors"]))
