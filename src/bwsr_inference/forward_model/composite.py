import numpy as np
import numpy.typing as npt

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal, VoigtSignal
from bwsr_inference.forward_model.background import PolynomialBackground


class H0_Polynomial(AbstractForwardModel):
    r"""
    Composite hypothesis for the null formulation (H0), asserting the presence 
    of exclusively a continuous stochastic background modeled via a polynomial.

    Theoretical Formulation:
        M_{H0}(f, \theta) = \sum_{i=0}^{N} \theta_i \cdot f^i
    """

    def __init__(self) -> None:
        """
        Initialize the H0 composite forward model.
        """
        self._background_model = PolynomialBackground()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the H0 deterministic model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The 1-dimensional array of independent frequency variables.
        theta : npt.NDArray[np.float64]
            The background parameter coefficients vector.

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated deterministic background array.

        Raises
        ------
        ValueError
            If the parameter vector is empty.
        """
        if theta.size < 1:
            raise ValueError("H0_Polynomial formulation requires at least one background coefficient.")
        
        return self._background_model(frequencies, theta)


class H1_Poly_Gaussian(AbstractForwardModel):
    r"""
    Composite hypothesis for the alternative formulation (H1), asserting the 
    presence of a latent Gaussian transient signal superimposed upon a 
    polynomial continuous background.

    Theoretical Formulation:
        M_{H1}(f, \theta) = B(f, \theta_{bg}) + S(f, \theta_{sig})
    """

    def __init__(self) -> None:
        """
        Initialize the H1 Gaussian composite forward model.
        """
        self._background_model = PolynomialBackground()
        self._signal_model = GaussianSignal()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the H1 Gaussian deterministic model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The 1-dimensional array of independent frequency variables.
        theta : npt.NDArray[np.float64]
            The concatenated parameter vector. The final three elements 
            strictly correspond to the Gaussian signal parameters 
            (Amplitude, Center, Width). All preceding elements map to 
            the polynomial background coefficients.

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated deterministic composite array.

        Raises
        ------
        ValueError
            If the parameter vector contains fewer than four elements.
        """
        if theta.size < 4:
            raise ValueError(f"H1_Poly_Gaussian formulation requires a minimum of 4 parameters. Received {theta.size}.")
        
        theta_bg = theta[:-3]
        theta_sig = theta[-3:]
        
        return self._background_model(frequencies, theta_bg) + self._signal_model(frequencies, theta_sig)


class H1_Poly_Voigt(AbstractForwardModel):
    r"""
    Composite hypothesis for the alternative formulation (H1), asserting the 
    presence of a latent pseudo-Voigt transient signal superimposed upon a 
    polynomial continuous background.

    Theoretical Formulation:
        M_{H1}(f, \theta) = B(f, \theta_{bg}) + V(f, \theta_{sig})
    """

    def __init__(self) -> None:
        """
        Initialize the H1 Voigt composite forward model.
        """
        self._background_model = PolynomialBackground()
        self._signal_model = VoigtSignal()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the H1 Voigt deterministic model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The 1-dimensional array of independent frequency variables.
        theta : npt.NDArray[np.float64]
            The concatenated parameter vector. The final four elements 
            strictly correspond to the Voigt signal parameters 
            (Amplitude, Center, Sigma, Gamma). All preceding elements 
            map to the polynomial background coefficients.

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated deterministic composite array.

        Raises
        ------
        ValueError
            If the parameter vector contains fewer than five elements.
        """
        if theta.size < 5:
            raise ValueError(f"H1_Poly_Voigt formulation requires a minimum of 5 parameters. Received {theta.size}.")
        
        theta_bg = theta[:-4]
        theta_sig = theta[-4:]
        
        return self._background_model(frequencies, theta_bg) + self._signal_model(frequencies, theta_sig)