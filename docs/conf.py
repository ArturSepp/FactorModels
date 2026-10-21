"""Build the public FactorModels documentation with MyST."""

project = "FactorModels"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
extensions = ["myst_parser"]
html_theme = "furo"
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]
