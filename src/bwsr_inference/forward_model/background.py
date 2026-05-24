import numpy as np
import numpy.typing as npt

# Corrected namespace
from bwsr_inference.forward_model.signal import AbstractForwardModel
from bwsr_inference.core import _core_models


class ExponentialBackground(AbstractForwardModel):
    r"""
    Parametric model for an exponentially decaying stochastic noise background.

    The model formulation follows:
    B(f, \theta) = \theta_0 \cdot \exp(-\theta_1 \cdot f) + \theta_2
    """

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the Exponential background model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Independent variable array.
        theta : npt.NDArray[np.float64]
            Parameter vector mapped as:
            theta[0] : Base amplitude (A)
            theta[1] : Rate of decay (\lambda)
            theta[2] : Baseline offset (C)

        Returns
        -------
        npt.NDArray[np.float64]
            Model evaluation array.
            
        Raises
        ------
        ValueError
            If the parameter vector contains an insufficient number of elements.
        """
        if theta.size != 3:
            raise ValueError(f"ExponentialBackground requires exactly 3 parameters. Received {theta.size}.")

        return _core_models.evaluate_exponential_background(
            frequencies,
            float(theta[0]),
            float(theta[1]),
            float(theta[2])
        )


class PolynomialBackground(AbstractForwardModel):
    r"""
    Parametric model for a polynomial continuous stochastic noise background.

    The model formulation follows:
    B(f, \theta) = \sum_{i=0}^{N} \theta_i \cdot f^i
    
    Evaluations utilize Horner's Method within the C++ layer to ensure numerical
    stability when modeling higher-order background variations across wide 
    frequency bandpasses.
    """

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the Polynomial background model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Independent variable array.
        theta : npt.NDArray[np.float64]
            Parameter vector mapped to polynomial coefficients, ordered from 
            lowest degree (constant term) to highest degree.
            theta = [c_0, c_1, c_2, ..., c_n]

        Returns
        -------
        npt.NDArray[np.float64]
            Model evaluation array.
            
        Raises
        ------
        ValueError
            If the parameter vector is empty.
        """
        if theta.size == 0:
            raise ValueError("PolynomialBackground requires at least one parameter coefficient.")

        return _core_models.evaluate_polynomial_background(frequencies, theta.astype(np.float64))