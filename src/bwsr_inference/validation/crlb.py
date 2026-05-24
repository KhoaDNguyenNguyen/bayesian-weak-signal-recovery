import pickle
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Tuple, Dict, Any


class FisherInformationMatrix:
    """
    Computes the Fisher Information Matrix (FIM) and the Cramér-Rao Lower 
    Bound (CRLB) for a given deterministic forward model and observational 
    noise configuration.

    The CRLB establishes the theoretical lower limit on the variance of any 
    unbiased estimator, calculated as the diagonal elements of the inverted FIM.

    Theoretical Formulation:
        I_{jk} = \\sum_{i=1}^{N} \\frac{1}{\\sigma_i^2} 
                 \\frac{\\partial P(f_i, \\theta)}{\\partial \\theta_j} 
                 \\frac{\\partial P(f_i, \\theta)}{\\partial \\theta_k}
        CRLB(\\theta_j) = (I^{-1})_{jj}
    """

    def __init__(
        self,
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        frequencies: npt.NDArray[np.float64],
        noise_variance: npt.NDArray[np.float64]
    ) -> None:
        """
        Initialize the FIM computation context.

        Parameters
        ----------
        forward_model : Callable
            The analytical function defining the parametric physical hypothesis.
        frequencies : npt.NDArray[np.float64]
            The 1-dimensional array of independent variables (e.g., frequencies).
        noise_variance : npt.NDArray[np.float64]
            The 1-dimensional array containing the observational variance for each datum.

        Raises
        ------
        ValueError
            If the independent variables and noise variance arrays exhibit mismatched 
            dimensions or if variance arrays contain non-positive values.
        """
        if frequencies.shape != noise_variance.shape:
            raise ValueError("Dimensionality mismatch between independent variables and noise variance.")
        if np.any(noise_variance <= 0.0):
            raise ValueError("Noise variance must be strictly positive definite.")

        self._forward_model = forward_model
        self._frequencies = frequencies
        self._inv_variance = 1.0 / noise_variance

    def _compute_jacobian(
        self, 
        theta: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the Jacobian matrix of the forward model with respect to the 
        parameters utilizing second-order central finite differences.

        Parameters
        ----------
        theta : npt.NDArray[np.float64]
            The parameter vector evaluated at the Maximum A Posteriori (MAP) estimate.

        Returns
        -------
        npt.NDArray[np.float64]
            A 2-dimensional array of shape (N, D), representing the Jacobian matrix, 
            where N is the number of observations and D is the number of parameters.
        """
        n_params = theta.size
        n_samples = self._frequencies.size
        jacobian = np.zeros((n_samples, n_params), dtype=np.float64)

        for j in range(n_params):
            delta = np.zeros_like(theta)
            # Implement an adaptive step size conditioned on parameter magnitude
            step = 1e-5 * np.maximum(np.abs(theta[j]), 1.0)
            delta[j] = step

            theta_plus = theta + delta
            theta_minus = theta - delta

            model_plus = self._forward_model(self._frequencies, theta_plus)
            model_minus = self._forward_model(self._frequencies, theta_minus)

            jacobian[:, j] = (model_plus - model_minus) / (2.0 * step)

        return jacobian

    def evaluate(
        self, 
        theta_map: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute the Fisher Information Matrix and subsequent Cramér-Rao Lower Bounds.

        Parameters
        ----------
        theta_map : npt.NDArray[np.float64]
            The Maximum A Posteriori parameter vector derived from Bayesian inference.

        Returns
        -------
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            - fim: The symmetric, positive-definite Fisher Information Matrix (D x D).
            - crlb: A 1-dimensional array of the theoretical minimum variance for each parameter (D,).

        Raises
        ------
        np.linalg.LinAlgError
            If the computed FIM is singular or ill-conditioned, preventing inversion.
        """
        jacobian = self._compute_jacobian(theta_map)
        
        # FIM Formulation: J^T * W * J where W is the diagonal precision matrix
        fim = np.dot(jacobian.T, self._inv_variance[:, np.newaxis] * jacobian)

        try:
            # Utilize pseudo-inverse for numerical stability on ill-conditioned boundaries
            fim_inverse = np.linalg.pinv(fim, rcond=1e-15)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Matrix inversion failure during CRLB computation: {e}")

        # Ensure diagonal values remain strictly positive to offset numerical truncation artifacts
        crlb = np.maximum(np.diag(fim_inverse), 1e-30)

        return fim, crlb


def extract_posterior_statistics(
    inference_results_path: Path
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Extract the Maximum A Posteriori (MAP) estimate and the empirical parameter 
    covariance from the serialized nested sampling outputs.

    Parameters
    ----------
    inference_results_path : Path
        Filesystem path to the serialized '.pkl' result from the inference engine.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        - theta_map: The parameter vector maximizing the log-likelihood.
        - empirical_variance: The diagonal of the weighted parameter covariance matrix.

    Raises
    ------
    IOError
        If the inference binary target is unreadable or malformed.
    """
    try:
        with open(inference_results_path, 'rb') as file_descriptor:
            results = pickle.load(file_descriptor)
    except Exception as e:
        raise IOError(f"Failed to ingest serialized inference payload: {e}")

    samples = results['samples']
    logl = results['log_likelihoods']
    
    # Derivation of strictly normalized importance weights
    unnormalized_weights = np.exp(results['weights'] - np.max(results['weights']))
    weights = unnormalized_weights / np.sum(unnormalized_weights)

    # MAP Estimate Formulation
    map_index = np.argmax(logl)
    theta_map = samples[map_index, :]

    # Weighted empirical covariance matrix derivation
    covariance_matrix = np.cov(samples.T, aweights=weights)
    empirical_variance = np.diag(covariance_matrix)

    return theta_map, empirical_variance


def generate_crlb_diagnostic_plot(
    empirical_variance: npt.NDArray[np.float64],
    theoretical_crlb: npt.NDArray[np.float64],
    output_path: Path
) -> None:
    """
    Synthesize a vector graphic overlaying the asymptotic theoretical limits 
    against the empirically derived variances to visually confirm 
    estimator efficiency.

    Parameters
    ----------
    empirical_variance : npt.NDArray[np.float64]
        The variance array obtained from Bayesian posterior samples.
    theoretical_crlb : npt.NDArray[np.float64]
        The strict minimum variance array computed via the FIM.
    output_path : Path
        The target path for the compiled vector graphic (.pdf).
    """
    n_params = empirical_variance.size
    indices = np.arange(n_params)
    width = 0.35

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        indices - width/2, 
        theoretical_crlb, 
        width, 
        label='Theoretical Lower Bound (CRLB)', 
        color='#1f77b4', 
        edgecolor='black'
    )
    ax.bar(
        indices + width/2, 
        empirical_variance, 
        width, 
        label='Empirical Variance (Posterior)', 
        color='#ff7f0e', 
        edgecolor='black'
    )

    ax.set_ylabel('Variance (Logarithmic Scale)')
    ax.set_xlabel('Parameter Coordinate Index')
    ax.set_title('Asymptotic Efficiency Constraint Validation')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"$\\theta_{{{i}}}$" for i in range(n_params)])
    ax.set_yscale('log')
    
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, format='pdf', dpi=300)
    except Exception as e:
        raise IOError(f"Failure during vector graphic compilation: {e}")
    finally:
        plt.close(fig)


def evaluate_asymptotic_limits(
    inference_results_path: Path,
    data_path: Path,
    forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    output_plot_path: Path
) -> Dict[str, Any]:
    """
    Orchestrate the extraction of inference results, computation of the FIM, 
    and compilation of the diagnostic evaluation verifying physical optimality.

    Parameters
    ----------
    inference_results_path : Path
        Path to the `.pkl` payload defining the multi-dimensional posterior.
    data_path : Path
        Path to the Layer 1 observational datum `.csv`.
    forward_model : Callable
        The deterministic analytical model representation.
    output_plot_path : Path
        Output path for the CRLB vector graphic diagnostic.

    Returns
    -------
    Dict[str, Any]
        A mapping containing the raw MAP vector, FIM, CRLB, and empirical statistics.
    """
    # 1. Ingest Observational Data (D_i)
    try:
        data_matrix = np.genfromtxt(data_path, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        noise_variance = data_matrix[:, 2]
    except Exception as e:
        raise IOError(f"Failed to ingest Layer 1 observational dataset: {e}")

    # 2. Extract Posterior Extrema and Covariance Statistics
    theta_map, empirical_variance = extract_posterior_statistics(inference_results_path)

    # 3. Compute Theoretical Boundaries (FIM / CRLB)
    fisher_engine = FisherInformationMatrix(
        forward_model=forward_model,
        frequencies=frequencies,
        noise_variance=noise_variance
    )
    fim, crlb = fisher_engine.evaluate(theta_map)

    # 4. Generate Output Validation Diagnostic Graphic
    generate_crlb_diagnostic_plot(
        empirical_variance=empirical_variance,
        theoretical_crlb=crlb,
        output_path=output_plot_path
    )

    return {
        "theta_map": theta_map,
        "empirical_variance": empirical_variance,
        "fisher_information_matrix": fim,
        "cramer_rao_lower_bound": crlb
    }