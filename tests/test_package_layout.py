"""Repository example names identify one runnable workflow each."""
from pathlib import Path


def test_example_basenames_are_unique():
    """Method demonstrations and complete workflows have distinct navigation names."""
    examples = Path(__file__).resolve().parents[1] / "examples"
    names = [p.name for p in examples.rglob("*.py") if p.name != "__init__.py"]
    assert len(names) == len(set(names)), names
