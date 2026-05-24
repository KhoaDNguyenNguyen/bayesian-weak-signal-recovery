"""
Asymptotic validation and hypothesis falsification diagnostics.

This module implements rigorous statistical falsification methodologies 
including Cramér-Rao Lower Bound computations, Look-Elsewhere Effect 
corrections, and structured residual diagnostics.
"""

from bwsr_inference.validation.crlb import FisherInformationMatrix, evaluate_asymptotic_limits
from bwsr_inference.validation.look_elsewhere import GlobalSignificanceEvaluator, SignificanceMetrics
from bwsr_inference.validation.residuals import ResidualDiagnostics

__all__ = [
    "FisherInformationMatrix",
    "evaluate_asymptotic_limits",
    "GlobalSignificanceEvaluator",
    "SignificanceMetrics",
    "ResidualDiagnostics"
]