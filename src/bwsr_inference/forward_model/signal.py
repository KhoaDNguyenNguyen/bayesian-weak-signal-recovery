import abc
import numpy as np
import numpy.typing as npt

# Corrected namespace from ssr_inference to bwsr_inference
from bwsr_inference.core import _core_models


class AbstractForwardModel(abc.ABC):
    r"""
    Abstract base class defining the operational interface for deterministic 
    forward models within the Bayesian inference pipeline.
    """
    
    @abc.abstractmethod
    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the parametric model over a set of independent variables.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            A 1-dimensional array of frequency bins.
        theta : npt.NDArray[np.float64]
            The parameter vector corresponding to a specific state within the 
            hypothesis space.

        Returns
        -------
        npt.NDArray[np.float64]
            A 1-dimensional array representing the predicted power spectral density.
        """
        pass


class GaussianSignal(AbstractForwardModel):
    r"""
    Parametric model for a Gaussian transient radio pulse.

    The model formulation follows:
    S(f, \theta) = \theta_0 \cdot \exp\left(-\frac{(f - \theta_1)^2}{2\theta_2^2}\right)
    """

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the Gaussian model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Independent variable array.
        theta : npt.NDArray[np.float64]
            Parameter vector mapped as:
            theta[0] : Amplitude (A)
            theta[1] : Center frequency (\mu)
            theta[2] : Standard deviation (\sigma)

        Returns
        -------
        npt.NDArray[np.float64]
            Model evaluation array.
            
        Raises
        ------
        ValueError
            If the parameter vector contains an insufficient number of elements.
        RuntimeError
            If unphysical parameter states are evaluated (e.g., non-positive \sigma).
        """
        if theta.size != 3:
            raise ValueError(f"GaussianSignal requires exactly 3 parameters. Received {theta.size}.")

        return _core_models.evaluate_gaussian(
            frequencies,
            float(theta[0]),
            float(theta[1]),
            float(theta[2])
        )


class VoigtSignal(AbstractForwardModel):
    r"""
    Parametric model for a Voigt profile radio transient.

    Implemented using the computationally efficient pseudo-Voigt formulation 
    based on the linear combination of Gaussian and Lorentzian functions,
    calibrated to the Whiting (1968) exact width mappings.
    """

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the pseudo-Voigt model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Independent variable array.
        theta : npt.NDArray[np.float64]
            Parameter vector mapped as:
            theta[0] : Amplitude (A)
            theta[1] : Center frequency (\mu)
            theta[2] : Gaussian standard deviation (\sigma)
            theta[3] : Lorentzian half-width at half-maximum (\gamma)

        Returns
        -------
        npt.NDArray[np.float64]
            Model evaluation array.
            
        Raises
        ------
        ValueError
            If the parameter vector contains an insufficient number of elements.
        RuntimeError
            If unphysical parameter states are evaluated.
        """
        if theta.size != 4:
            raise ValueError(f"VoigtSignal requires exactly 4 parameters. Received {theta.size}.")

        return _core_models.evaluate_pseudo_voigt(
            frequencies,
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3])
        )