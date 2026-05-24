#!/usr/bin/env python3
import argparse
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.typing as npt
from typing import Tuple

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal
from bwsr_inference.forward_model.background import ExponentialBackground
from bwsr_inference.validation.residuals import ResidualDiagnostics


class CompositeForwardModel(AbstractForwardModel):
    r"""
    Composite deterministic model integrating an exponential stochastic 
    background with a transient Gaussian signal hypothesis.

    Theoretical Formulation:
        M(f, \theta) = B(f, \theta_{0:3}) + S(f, \theta_{3:6})
    """

    def __init__(self) -> None:
        """Initialize the constituent background and transient models."""
        self._background_model = ExponentialBackground()
        self._signal_model = GaussianSignal()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Evaluate the composite forward model.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            The 1-dimensional independent variable array.
        theta : npt.NDArray[np.float64]
            The concatenated parameter vector (length 6).

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated deterministic model array.
        """
        if theta.size != 6:
            raise ValueError(f"CompositeForwardModel requires exactly 6 parameters. Received {theta.size}.")

        return self._background_model(frequencies, theta[0:3]) + \
               self._signal_model(frequencies, theta[3:6])


def parse_arguments() -> argparse.Namespace:
    r"""
    Parse command-line arguments for executing the diagnostic evaluation phase.
    """
    parser = argparse.ArgumentParser(
        description="Assess goodness-of-fit and isolate systematic residual structures via LOWESS."
    )
    parser.add_argument(
        '--data', 
        type=Path, 
        required=True,
        help="Path to the Layer 1 observational data CSV."
    )
    parser.add_argument(
        '--results', 
        type=Path, 
        required=True,
        help="Path to the serialized Nested Sampling inference payload (.pkl)."
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path("data/processed/residual_lowess.pdf"),
        help="Target output path for the diagnostic vector graphic."
    )
    parser.add_argument(
        '--frac', 
        type=float, 
        default=0.15,
        help="Fractional bandwidth for the LOWESS local regression."
    )
    return parser.parse_args()


def extract_posterior_statistics(results_path: Path) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    
    r"""
    Extract the Expected Value (Posterior Mean) and Standard Deviation 
    from the weighted samples of the serialized dynesty Results object.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        The posterior mean vector and the posterior standard deviation vector.
    """
    try:
        with open(results_path, 'rb') as file_descriptor:
            inference_data = pickle.load(file_descriptor)
            
        samples = inference_data['samples']
        log_weights = inference_data['logwt']
        
        # Convert log-weights to normalized linear weights.
        # Subtraction of the maximum log-weight ensures numerical stability 
        # against floating-point overflow during exponentiation.
        unnormalized_weights = np.exp(log_weights - np.max(log_weights))
        normalized_weights = unnormalized_weights / np.sum(unnormalized_weights)
        
        # Calculate the marginalized posterior mean: E[\theta]
        theta_mean = np.average(samples, weights=normalized_weights, axis=0)
        
        # Calculate the marginalized posterior standard deviation: \sqrt{Var[\theta]}
        theta_variance = np.average((samples - theta_mean)**2, weights=normalized_weights, axis=0)
        theta_std = np.sqrt(theta_variance)
        
        return theta_mean, theta_std
    except Exception as e:
        raise IOError(f"Failed to extract statistical estimates from {results_path}: {e}")

def generate_diagnostic_graphic(
    frequencies: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    residuals: np.ndarray,
    lowess_trend: np.ndarray,
    chi_squared_nu: float,
    output_path: Path
) -> None:
    r"""
    Compile a high-resolution, academic-grade vector graphic utilizing 
    the native LaTeX rendering engine for publication-quality typography.

    Parameters
    ----------
    frequencies : np.ndarray
        The independent variable array.
    data : np.ndarray
        The observational data array.
    model : np.ndarray
        The evaluated model prediction array.
    residuals : np.ndarray
        The raw residual array.
    lowess_trend : np.ndarray
        The non-parametric structural trend.
    chi_squared_nu : float
        The reduced chi-squared statistical scalar.
    output_path : Path
        Target file system path.
    """
    # Enforce strict LaTeX typography rules
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
    })

    fig, (ax_main, ax_res) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(10, 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )

    ax_main.scatter(frequencies, data, s=2, color='black', alpha=0.5, label=r'Observational Data')
    ax_main.plot(frequencies, model, color='#d62728', linewidth=2, label=r'Posterior Mean Prediction')
    ax_main.set_ylabel(r'Power Spectral Density $[\mathrm{W/Hz}]$')
    
    # Mathematical formatting utilizing LaTeX
    title_string = rf'Diagnostic Falsification Analysis ($\chi^2/\nu = {chi_squared_nu:.3f}$)'
    ax_main.set_title(title_string)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, which="both", ls="--", alpha=0.3)

    ax_res.scatter(frequencies, residuals, s=2, color='gray', alpha=0.6, label=r'Raw Residuals')
    ax_res.axhline(0.0, color='black', linewidth=1, linestyle='--')
    ax_res.plot(frequencies, lowess_trend, color='#1f77b4', linewidth=2.5, label=r'LOWESS Systematic Trend')
    ax_res.set_ylabel(r'Residuals $[\mathrm{W/Hz}]$')
    ax_res.set_xlabel(r'Frequency $[\mathrm{Hz}]$')
    ax_res.legend(loc='upper right')
    ax_res.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure bbox_inches='tight' is passed to prevent LaTeX bounding box clipping
        fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    except Exception as e:
        raise IOError(f"Failure during diagnostic vector graphic compilation (Verify local TeX distribution): {e}")
    finally:
        plt.close(fig)

def main() -> None:
    r"""
    Primary execution sequence for model falsification via residual diagnostics.
    """
    args = parse_arguments()

    if not args.data.is_file():
        sys.stderr.write(f"Error: Observational data array {args.data} cannot be located.\n")
        sys.exit(1)

    if not args.results.is_file():
        sys.stderr.write(f"Error: Inference payload {args.results} cannot be located.\n")
        sys.exit(1)

    try:
        data_matrix = np.genfromtxt(args.data, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        observational_data = data_matrix[:, 1]
        noise_variance = data_matrix[:, 2]

        theta_mean, theta_std = extract_posterior_statistics(args.results)
        
        forward_model = CompositeForwardModel()
        model_prediction = forward_model(frequencies, theta_mean)

        diagnostics = ResidualDiagnostics(
            observational_data=observational_data,
            model_prediction=model_prediction,
            noise_variance=noise_variance,
            n_parameters=theta_mean.size
        )

        chi_squared_nu = diagnostics.compute_reduced_chi_squared()
        lowess_trend = diagnostics.compute_lowess_trend(frequencies, fraction=args.frac)
        
        sys.stdout.write("Diagnostic evaluation successfully executed.\n")
        sys.stdout.write("--- Extracted Posterior Statistics ---\n")
        
        labels = ["BG_Amp", "BG_Decay", "BG_Offset", "Sig_Amp", "Sig_Center", "Sig_Width"]
        for label, mean, std in zip(labels, theta_mean, theta_std):
            sys.stdout.write(f"{label:<15}: {mean:>10.5f} +/- {std:>10.5f}\n")
            
        sys.stdout.write(f"\nReduced Chi-Squared (Goodness-of-Fit): {chi_squared_nu:.5f}\n")

        generate_diagnostic_graphic(
            frequencies=frequencies,
            data=observational_data,
            model=model_prediction,
            residuals=diagnostics.raw_residuals,
            lowess_trend=lowess_trend,
            chi_squared_nu=chi_squared_nu,
            output_path=args.output
        )
        sys.stdout.write(f"Validation graphic synthesized at: {args.output}\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to underlying numerical or IO error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()