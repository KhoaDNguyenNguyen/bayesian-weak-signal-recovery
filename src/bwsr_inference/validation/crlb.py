import shutil
import warnings
import pickle
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Tuple, Dict, Any


class FisherInformationMatrix:
    r"""
    Computes the Fisher Information Matrix (FIM) and the Cramér-Rao Lower 
    Bound (CRLB) for a given deterministic forward model and observational 
    noise configuration.

    The CRLB establishes the theoretical lower limit on the variance of any 
    unbiased estimator, calculated as the diagonal elements of the inverted FIM.

    Theoretical Formulation:
        I_{jk} = \sum_{i=1}^{N} \frac{1}{\sigma_i^2} 
                 \frac{\partial P(f_i, \theta)}{\partial \theta_j} 
                 \frac{\partial P(f_i, \theta)}{\partial \theta_k}
        CRLB(\theta_j) = (I^{-1})_{jj}
    """

    def __init__(
        self,
        forward_model: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        frequencies: npt.NDArray[np.float64],
        noise_variance: npt.NDArray[np.float64]
    ) -> None:
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
        n_params = theta.size
        n_samples = self._frequencies.size
        jacobian = np.zeros((n_samples, n_params), dtype=np.float64)

        for j in range(n_params):
            delta = np.zeros_like(theta)
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
        jacobian = self._compute_jacobian(theta_map)
        fim = np.dot(jacobian.T, self._inv_variance[:, np.newaxis] * jacobian)
        
        condition_number = np.linalg.cond(fim)
        if condition_number > 1e15:
            warnings.warn(
                f"Fisher Information Matrix is heavily ill-conditioned (Condition Number: {condition_number:.2e}). "
                "Parameter estimates may be degenerate. Proceeding with pseudo-inverse calculation.",
                category=RuntimeWarning
            )

        try:
            fim_inverse = np.linalg.pinv(fim, rcond=1e-15)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Matrix inversion failure during CRLB computation: {e}")

        crlb = np.maximum(np.diag(fim_inverse), 1e-30)

        return fim, crlb


def extract_posterior_statistics(
    inference_results_path: Path
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    try:
        with open(inference_results_path, 'rb') as file_descriptor:
            results = pickle.load(file_descriptor)
    except Exception as e:
        raise IOError(f"Failed to ingest serialized inference payload: {e}")

    samples = results['samples']
    # Correctly reference the internal dynesty log-likelihood key
    logl = results['logl']
    
    # unnormalized_weights = np.exp(results['weights'] - np.max(results['weights']))
    unnormalized_weights = np.exp(results['logwt'] - np.max(results['logwt']))
    weights = unnormalized_weights / np.sum(unnormalized_weights)

    map_index = np.argmax(logl)
    theta_map = samples[map_index, :]

    covariance_matrix = np.cov(samples.T, aweights=weights)
    empirical_variance = np.diag(covariance_matrix)

    return theta_map, empirical_variance


def generate_crlb_diagnostic_plot(
    empirical_variance: npt.NDArray[np.float64],
    theoretical_crlb: npt.NDArray[np.float64],
    output_path: Path
) -> None:
    n_params = empirical_variance.size
    indices = np.arange(n_params)
    width = 0.35

    use_tex = shutil.which("latex") is not None
    if not use_tex:
        warnings.warn("LaTeX distribution not detected. Defaulting to standard matplotlib text rendering capabilities.", category=UserWarning)

    plt.rcParams.update({
        "text.usetex": use_tex,  
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if use_tex else ["DejaVu Serif"],
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
    ax.set_xticklabels([fr"$\theta_{{{i}}}$" for i in range(n_params)])
    ax.set_yscale('log')
    
    ax.legend(loc='upper left')
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
    try:
        data_matrix = np.genfromtxt(data_path, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        noise_variance = data_matrix[:, 2]
    except Exception as e:
        raise IOError(f"Failed to ingest Layer 1 observational dataset: {e}")

    theta_map, empirical_variance = extract_posterior_statistics(inference_results_path)

    fisher_engine = FisherInformationMatrix(
        forward_model=forward_model,
        frequencies=frequencies,
        noise_variance=noise_variance
    )
    fim, crlb = fisher_engine.evaluate(theta_map)

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