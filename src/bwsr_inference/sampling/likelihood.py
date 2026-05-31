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
    Log-likelihood formulation assuming independent Gaussian noise.

    Theoretical Formulation:
    To guarantee numerical stability against instrumental over-filtering 
    (which drives \\sigma_i^2 \\to 0 in filter stopbands), an asymptotic 
    variance floor is rigorously enforced:
        \\sigma_{i, eff}^2 = \\max(\\sigma_i^2, \\epsilon \\cdot \\text{median}(\\sigma^2))
    
    The stabilized log-likelihood is:
        \\ln \\mathcal{L}(\\theta) = -\\frac{1}{2} \\sum_{i=1}^{N} \\left[ 
            \\frac{(D_i - P(f_i, \\theta))^2}{\\sigma_{i, eff}^2} + \\ln(2\\pi\\sigma_{i, eff}^2) 
        \\right]
    """

    def __init__(
        self,
        frequencies: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
        noise_variance: npt.NDArray[np.float64],
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        variance_floor_epsilon: float = 1e-5
    ) -> None:
        """
        Initialize the robust likelihood formulation and precompute invariants.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The independent variable array (f_i).
        data : npt.NDArray[np.float64]
            The observational data array (D_i).
        noise_variance : npt.NDArray[np.float64]
            The raw empirical variance associated with each datum (\\sigma_i^2).
        forward_model : Callable
            The deterministic mapping function.
        variance_floor_epsilon : float, optional
            The fractional scaling constant utilized to establish the minimum 
            allowable numerical variance. Default is 1e-5.
        """
        if not (frequencies.shape == data.shape == noise_variance.shape):
            raise ValueError("Dimensionality mismatch among independent variables, observational data, and noise arrays.")
        if np.any(noise_variance <= 0.0):
            raise ValueError("Noise variance must be strictly positive definite.")

        self._frequencies = frequencies
        self._data = data
        self._forward_model = forward_model

        # Execute Variance Regularization to prevent Inverse-Variance Explosion
        median_variance = np.median(noise_variance)
        absolute_variance_floor = median_variance * variance_floor_epsilon
        
        effective_variance = np.maximum(noise_variance, absolute_variance_floor)
        self._inv_variance = 1.0 / effective_variance

        n_samples = data.size
        self._normalization_constant = -0.5 * (
            n_samples * np.log(2.0 * np.pi) + np.sum(np.log(effective_variance))
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """
        Evaluate the variance-regularized diagonal Gaussian log-likelihood.
        """
        model_prediction = self._forward_model(self._frequencies, theta)
        residuals = self._data - model_prediction
        
        chi_squared = np.sum(np.square(residuals) * self._inv_variance)
        
        return float(self._normalization_constant - 0.5 * chi_squared)


class DenseGaussianLikelihood(AbstractLikelihood):
    """
    Log-likelihood formulation assuming correlated Gaussian noise characterized 
    by a dense covariance matrix.

    Incorporates Tikhonov (Ridge) regularization to ensure strict positive-definiteness 
    and avert ill-conditioned Cholesky decompositions during inversion.
    """

    def __init__(
        self,
        frequencies: npt.NDArray[np.float64],
        data: npt.NDArray[np.float64],
        covariance_matrix: npt.NDArray[np.float64],
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        variance_floor_epsilon: float = 1e-5
    ) -> None:
        """
        Initialize the stabilized dense likelihood utilizing Cholesky decomposition.
        """
        n_samples = data.size
        if covariance_matrix.shape != (n_samples, n_samples):
            raise ValueError(f"Covariance matrix must be of shape ({n_samples}, {n_samples}).")
        if frequencies.shape != data.shape:
            raise ValueError("Mismatch between independent variables and data arrays.")

        self._frequencies = frequencies
        self._data = data
        self._forward_model = forward_model

        # Execute Tikhonov Regularization
        median_diag = np.median(np.diag(covariance_matrix))
        regularization_matrix = np.eye(n_samples) * (median_diag * variance_floor_epsilon)
        effective_covariance = covariance_matrix + regularization_matrix

        try:
            chol_L = np.linalg.cholesky(effective_covariance)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Effective covariance matrix remains non-positive definite after regularization: {e}")
        
        # Precompute the precision matrix (Inverse Covariance)
        identity = np.eye(n_samples)
        chol_inv = np.linalg.solve(chol_L, identity)
        self._inv_covariance = np.dot(chol_inv.T, chol_inv)

        # Precompute log-determinant
        log_det_C = 2.0 * np.sum(np.log(np.diag(chol_L)))
        self._normalization_constant = -0.5 * (
            n_samples * np.log(2.0 * np.pi) + log_det_C
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """
        Evaluate the regularized dense multivariate Gaussian log-likelihood.
        """
        model_prediction = self._forward_model(self._frequencies, theta)
        residuals = self._data - model_prediction
        
        chi_squared = np.dot(residuals, np.dot(self._inv_covariance, residuals))
        
        return float(self._normalization_constant - 0.5 * chi_squared)