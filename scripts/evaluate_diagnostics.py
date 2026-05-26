#!/usr/bin/env python3
import argparse
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.typing as npt
from typing import Tuple, List

from bwsr_inference.forward_model import MODEL_ZOO
from bwsr_inference.validation.residuals import ResidualDiagnostics

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual diagnostics and Goodness-of-Fit.")
    parser.add_argument('--data', type=Path, required=True, help="Path to the Layer 1 observational CSV data.")
    parser.add_argument('--results', type=Path, required=True, help="Path to the serialized Nested Sampling inference results.")
    parser.add_argument('--output', type=Path, required=True, help="Target output path for the diagnostic graphic (.pdf).")
    parser.add_argument('--model', type=str, choices=list(MODEL_ZOO.keys()), required=True, help="Hypothesis classification to dictate the forward model construction via the MODEL_ZOO.")
    parser.add_argument('--frac', type=float, default=0.15, help="Bandwidth fraction for the LOWESS non-parametric structural trend fitting.")
    return parser.parse_args()

def extract_posterior_statistics(results_path: Path) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    samples = data['samples']
    log_weights = data['logwt']
    unnormalized_weights = np.exp(log_weights - np.max(log_weights))
    normalized_weights = unnormalized_weights / np.sum(unnormalized_weights)
    theta_mean = np.average(samples, weights=normalized_weights, axis=0)
    theta_variance = np.average((samples - theta_mean)**2, weights=normalized_weights, axis=0)
    return theta_mean, np.sqrt(theta_variance)

def _resolve_parameter_labels(model_id: str, total_parameters: int) -> List[str]:
    """
    Dynamically resolve parameter naming conventions to support arbitrary background complexity.
    """
    if model_id == 'h0':
        n_bg = total_parameters
        sig_labels: List[str] = []
    elif model_id == 'h1_gauss':
        n_bg = total_parameters - 3
        sig_labels = ["Sig_Amp", "Sig_Center", "Sig_Width"]
    elif model_id == 'h1_voigt':
        n_bg = total_parameters - 4
        sig_labels = ["Sig_Amp", "Sig_Center", "Sig_Sigma", "Sig_Gamma"]
    else:
        return [f"Theta_{i}" for i in range(total_parameters)]

    if n_bg == 1:
        bg_labels = ["AWGN_Floor"]
    else:
        bg_labels = [f"Bg_Coeff_{i}" for i in range(n_bg)]
        
    return bg_labels + sig_labels

def generate_diagnostic_graphic(frequencies, data, model, residuals, lowess_trend, chi_squared_nu, output_path):
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12, "font.size": 10, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10
    })

    fig, (ax_main, ax_res) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax_main.scatter(frequencies, data, s=2, color='black', alpha=0.5, label=r'Observational Data')
    ax_main.plot(frequencies, model, color='#d62728', linewidth=2, label=r'Posterior Mean Prediction')
    ax_main.set_ylabel(r'Power Spectral Density $[\mathrm{W/Hz}]$')
    ax_main.set_title(rf'Diagnostic Falsification Analysis ($\chi^2/\nu = {chi_squared_nu:.3f}$)')
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
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    args = parse_arguments()
    data_matrix = np.genfromtxt(args.data, delimiter=',', skip_header=1)
    frequencies = data_matrix[:, 0]
    observational_data = data_matrix[:, 1]
    noise_variance = data_matrix[:, 2]

    theta_mean, theta_std = extract_posterior_statistics(args.results)
    
    forward_model = MODEL_ZOO[args.model]
    labels = _resolve_parameter_labels(args.model, theta_mean.size)

    model_prediction = forward_model(frequencies, theta_mean)

    diagnostics = ResidualDiagnostics(
        observational_data=observational_data, model_prediction=model_prediction,
        noise_variance=noise_variance, n_parameters=theta_mean.size
    )

    chi_squared_nu = diagnostics.compute_reduced_chi_squared()
    lowess_trend = diagnostics.compute_lowess_trend(frequencies, fraction=args.frac)
    
    sys.stdout.write("--- Extracted Posterior Statistics ---\n")
    for label, mean, std in zip(labels, theta_mean, theta_std):
        sys.stdout.write(f"{label:<15}: {mean:>15.8e} +/- {std:>15.8e}\n")
        
    sys.stdout.write(f"\nReduced Chi-Squared: {chi_squared_nu:.5f}\n")

    generate_diagnostic_graphic(
        frequencies, observational_data, model_prediction,
        diagnostics.raw_residuals, lowess_trend, chi_squared_nu, args.output
    )

if __name__ == "__main__":
    main()