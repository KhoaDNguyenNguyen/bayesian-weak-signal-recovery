import abc
import numpy as np
import numpy.typing as npt
from typing import Callable


class PriorTransform:
    """
    Constructs the deterministic mapping from the unit hypercube to the 
    physical parameter space for Bayesian inference.

    The transformation maps a vector of uniformly distributed random variables 
    u in [0, 1]^D to the physical parameter vector \\theta in R^D, conforming 
    to the defined prior boundaries and scale assumptions (linear or logarithmic).

    Theoretical Formulations:
    Linear (Uniform) Prior:
        \\theta_i = a_i + u_i \\cdot (b_i - a_i)
    Log-Uniform (Jeffreys) Prior:
        \\theta_i = \\exp( \\ln(a_i) + u_i \\cdot (\\ln(b_i) - \\ln(a_i)) )
    """

    def __init__(
        self, 
        bounds: npt.NDArray[np.float64], 
        log_flags: npt.NDArray[np.bool_]
    ) -> None:
        """
        Initialize the prior transformation matrix and precompute constants.

        Parameters
        ----------
        bounds : npt.NDArray[np.float64]
            A 2-dimensional array of shape (D, 2) where D is the dimensionality 
            of the parameter space. Each row specifies the [lower, upper] bounds.
        log_flags : npt.NDArray[np.bool_]
            A 1-dimensional boolean array of shape (D,) indicating whether 
            the corresponding parameter requires a log-uniform transformation.

        Raises
        ------
        ValueError
            If dimensionalities do not match, if lower bounds are not strictly 
            less than upper bounds, or if log-uniform priors are requested 
            for non-positive domains.
        """
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must be a 2-dimensional array of shape (D, 2).")
        if bounds.shape[0] != log_flags.shape[0]:
            raise ValueError("The dimension of log_flags must match the number of parameters (D).")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Lower bounds must be strictly less than upper bounds.")
        if np.any(log_flags & (bounds[:, 0] <= 0.0)):
            raise ValueError("Log-uniform distributions require strictly positive physical boundaries.")

        self._linear_mask = ~log_flags
        self._log_mask = log_flags

        self._lower = bounds[:, 0]
        self._delta = bounds[:, 1] - bounds[:, 0]

        self._log_lower = np.zeros_like(self._lower)
        self._log_delta = np.zeros_like(self._delta)

        if np.any(self._log_mask):
            self._log_lower[self._log_mask] = np.log(bounds[self._log_mask, 0])
            self._log_delta[self._log_mask] = (
                np.log(bounds[self._log_mask, 1]) - self._log_lower[self._log_mask]
            )

    def __call__(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Execute the parameter space transformation.

        Parameters
        ----------
        u : npt.NDArray[np.float64]
            A 1-dimensional array of coordinates situated in the unit hypercube.

        Returns
        -------
        npt.NDArray[np.float64]
            The physical parameter vector \\theta.
        """
        theta = np.empty_like(u, dtype=np.float64)

        if np.any(self._linear_mask):
            theta[self._linear_mask] = (
                self._lower[self._linear_mask] + 
                u[self._linear_mask] * self._delta[self._linear_mask]
            )

        if np.any(self._log_mask):
            theta[self._log_mask] = np.exp(
                self._log_lower[self._log_mask] + 
                u[self._log_mask] * self._log_delta[self._log_mask]
            )

        return theta


class AbstractLikelihood(abc.ABC):
    """
    Abstract base class defining the rigorous interface for Likelihood formulations.
    """

    @abc.abstractmethod
    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """
        Evaluate the natural logarithm of the likelihood function.

        Parameters
        ----------
        theta : npt.NDArray[np.float64]
            The parameter vector corresponding to a specific state within the 
            hypothesis space.

        Returns
        -------
        float
            The computed log-likelihood scalar value.
        """
        pass


class DiagonalGaussianLikelihood(AbstractLikelihood):
    """
    Log-likelihood formulation assuming independent, non-stationary Gaussian 
    noise across the observational domain (diagonal covariance matrix).

    Theoretical Formulation:
        \\ln \\mathcal{L}(\\theta) = -\\frac{1}{2} \\sum_{i=1}^{N} \\left[ 
            \\frac{(D_i - P(f_i, \\theta))^2}{\\sigma_i^2} + \\ln(2\\pi\\sigma_i^2) 
        \\right]
    """

    def __init__(
        self,
        frequencies: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
        noise_variance: npt.NDArray[np.float64],
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> None:
        """
        Initialize the likelihood and precompute invariants to guarantee 
        computational efficiency during the sampling regime.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The independent variable array (f_i).
        data : npt.NDArray[np.float64]
            The observational data array (D_i).
        noise_variance : npt.NDArray[np.float64]
            The variance associated with each datum (\\sigma_i^2).
        forward_model : Callable
            The deterministic mapping function P(f_i, \\theta) defined in Chunk 3.1.

        Raises
        ------
        ValueError
            If array dimensions mismatch or if variance assumes unphysical 
            (non-positive) states.
        """
        if not (frequencies.shape == data.shape == noise_variance.shape):
            raise ValueError("Dimensionality mismatch among independent variables, observational data, and noise arrays.")
        if np.any(noise_variance <= 0.0):
            raise ValueError("Noise variance must be strictly positive definite.")

        self._frequencies = frequencies
        self._data = data
        self._inv_variance = 1.0 / noise_variance
        self._forward_model = forward_model

        n_samples = data.size
        self._normalization_constant = -0.5 * (
            n_samples * np.log(2.0 * np.pi) + np.sum(np.log(noise_variance))
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """
        Evaluate the diagonal Gaussian log-likelihood.
        """
        model_prediction = self._forward_model(self._frequencies, theta)
        residuals = self._data - model_prediction
        
        chi_squared = np.sum(np.square(residuals) * self._inv_variance)
        
        return float(self._normalization_constant - 0.5 * chi_squared)


class DenseGaussianLikelihood(AbstractLikelihood):
    """
    Log-likelihood formulation assuming correlated Gaussian noise characterized 
    by a dense covariance matrix.

    Theoretical Formulation:
        \\ln \\mathcal{L}(\\theta) = -\\frac{1}{2} \\left[ 
            \\mathbf{r}^T \\mathbf{C}^{-1} \\mathbf{r} + \\ln(|2\\pi\\mathbf{C}|) 
        \\right]
    where \\mathbf{r} = \\mathbf{D} - \\mathbf{P}(\\mathbf{f}, \\theta).
    """

    def __init__(
        self,
        frequencies: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
        covariance_matrix: npt.NDArray[np.float64],
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> None:
        """
        Initialize the likelihood and precompute matrix inversions utilizing 
        Cholesky decomposition for numerical stability.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The independent variable array.
        data : npt.NDArray[np.float64]
            The observational data array.
        covariance_matrix : npt.NDArray[np.float64]
            The symmetric, positive-definite noise covariance matrix (\\mathbf{C}).
        forward_model : Callable
            The deterministic mapping function.

        Raises
        ------
        ValueError
            If matrix dimensions are incompatible.
        np.linalg.LinAlgError
            If the covariance matrix is not positive-definite.
        """
        n_samples = data.size
        if covariance_matrix.shape != (n_samples, n_samples):
            raise ValueError(f"Covariance matrix must be of shape ({n_samples}, {n_samples}).")
        if frequencies.shape != data.shape:
            raise ValueError("Mismatch between independent variables and data arrays.")

        self._frequencies = frequencies
        self._data = data
        self._forward_model = forward_model

        # Extract Cholesky decomposition: C = L L^T
        chol_L = np.linalg.cholesky(covariance_matrix)
        
        # Precompute the inverse covariance matrix (Precision Matrix)
        identity = np.eye(n_samples)
        chol_inv = np.linalg.solve(chol_L, identity)
        self._inv_covariance = np.dot(chol_inv.T, chol_inv)

        # Precompute the log determinant: ln(|C|) = 2 * sum(ln(diag(L)))
        log_det_C = 2.0 * np.sum(np.log(np.diag(chol_L)))
        self._normalization_constant = -0.5 * (
            n_samples * np.log(2.0 * np.pi) + log_det_C
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """
        Evaluate the dense multivariate Gaussian log-likelihood.
        """
        model_prediction = self._forward_model(self._frequencies, theta)
        residuals = self._data - model_prediction
        
        # Evaluates r^T C^{-1} r
        chi_squared = np.dot(residuals, np.dot(self._inv_covariance, residuals))
        
        return float(self._normalization_constant - 0.5 * chi_squared)