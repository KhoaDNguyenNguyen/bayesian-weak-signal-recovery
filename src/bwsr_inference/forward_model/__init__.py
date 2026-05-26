"""
Deterministic parametric hypotheses and forward model formulations.

This module constructs the physical models mapped to the independent 
variables, capturing both the transient signal hypotheses and the 
stochastic background continua. It exposes a unified MODEL_ZOO for 
dynamic architectural abstraction and model selection.
"""

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal, VoigtSignal
from bwsr_inference.forward_model.background import ExponentialBackground, PolynomialBackground
from bwsr_inference.forward_model.composite import H0_Polynomial, H1_Poly_Gaussian, H1_Poly_Voigt

# Registry for dynamic hypothesis loading and testing
MODEL_ZOO = {
    'h0': H0_Polynomial(),
    'h1_gauss': H1_Poly_Gaussian(),
    'h1_voigt': H1_Poly_Voigt()
}

__all__ = [
    "AbstractForwardModel",
    "GaussianSignal",
    "VoigtSignal",
    "ExponentialBackground",
    "PolynomialBackground",
    "H0_Polynomial",
    "H1_Poly_Gaussian",
    "H1_Poly_Voigt",
    "MODEL_ZOO"
]