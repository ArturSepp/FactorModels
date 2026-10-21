"""Completed capital-market-assumption diagnostics and conditional inference."""

from factormodels.cma.diagnostics import (
    AdmissionPolicy,
    CmaDiagnosticInputs,
    CmaDiagnosticResult,
    CmaDiagnosticSpec,
    compute_cma_diagnostics,
    without_admitted_alpha,
)
from factormodels.cma.identification import (
    CmaIdentificationResult,
    compute_cma_identification,
)
from factormodels.cma.moments import annual_arithmetic_from_log
from factormodels.cma.uncertainty import (
    CmaConditionalGlsResult,
    compute_conditional_gls_intervals,
)

__all__ = [
    'AdmissionPolicy', 'CmaDiagnosticInputs', 'CmaDiagnosticResult', 'CmaDiagnosticSpec',
    'compute_cma_diagnostics', 'without_admitted_alpha', 'CmaIdentificationResult',
    'compute_cma_identification', 'CmaConditionalGlsResult', 'compute_conditional_gls_intervals',
    'annual_arithmetic_from_log',
]
