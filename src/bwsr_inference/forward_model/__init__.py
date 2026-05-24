"""
Deterministic parametric hypotheses and forward model formulations.

This module constructs the physical models mapped to the independent 
variables, capturing both the transient signal hypotheses and the 
stochastic background continua.
"""

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal, VoigtSignal
from bwsr_inference.forward_model.background import ExponentialBackground, PolynomialBackground

__all__ = [
    "AbstractForwardModel",
    "GaussianSignal",
    "VoigtSignal",
    "ExponentialBackground",
    "PolynomialBackground"
]