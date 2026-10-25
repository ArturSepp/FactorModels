"""Build the public FactorModels documentation with MyST."""

project = "FactorModels"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
extensions = ["myst_parser"]
html_theme = "furo"
html_title = "factormodels - factor-model construction, estimation and GLS analytics"
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]
