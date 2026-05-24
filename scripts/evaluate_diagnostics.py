#!/usr/bin/env python3
import argparse
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bwsr_inference.forward_model.signal import GaussianSignal
from bwsr_inference.validation.residuals import ResidualDiagnostics


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
        help="Path to the Layer 1 observational data CSV (e.g., baseline_spectrum.csv)."
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


def extract_map_estimate(results_path: Path) -> np.ndarray:
    r"""
    Extract the Maximum A Posteriori (MAP) parameter array from the serialized binary.
    """
    try:
        with open(results_path, 'rb') as file_descriptor:
            inference_data = pickle.load(file_descriptor)
            
        samples = inference_data['samples']
        # Correctly reference the internal dynesty log-likelihood key
        log_likelihoods = inference_data['logl']
        
        map_index = np.argmax(log_likelihoods)
        return samples[map_index, :]
    except Exception as e:
        raise IOError(f"Failed to extract MAP estimate from {results_path}: {e}")


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
    Compile a high-resolution, academic-grade vector graphic overlaying the 
    observational data against the MAP model, and analyzing the residual structures.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, (ax_main, ax_res) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(10, 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )

    ax_main.scatter(frequencies, data, s=2, color='black', alpha=0.5, label='Observational Data')
    ax_main.plot(frequencies, model, color='#d62728', linewidth=2, label='MAP Model Prediction')
    ax_main.set_ylabel('Power Spectral Density')
    ax_main.set_title(rf'Diagnostic Falsification Analysis ($\chi^2/\nu = {chi_squared_nu:.3f}$)')
    ax_main.legend(loc='upper right')
    ax_main.grid(True, which="both", ls="--", alpha=0.3)

    ax_res.scatter(frequencies, residuals, s=2, color='gray', alpha=0.6, label='Raw Residuals')
    ax_res.axhline(0.0, color='black', linewidth=1, linestyle='--')
    ax_res.plot(frequencies, lowess_trend, color='#1f77b4', linewidth=2.5, label='LOWESS Systematic Trend')
    ax_res.set_ylabel('Residuals')
    ax_res.set_xlabel('Frequency [Hz]')
    ax_res.legend(loc='upper right')
    ax_res.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='pdf', dpi=300)
    except Exception as e:
        raise IOError(f"Failure during diagnostic vector graphic compilation: {e}")
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

        theta_map = extract_map_estimate(args.results)
        
        forward_model = GaussianSignal()
        model_prediction = forward_model(frequencies, theta_map)

        diagnostics = ResidualDiagnostics(
            observational_data=observational_data,
            model_prediction=model_prediction,
            noise_variance=noise_variance,
            n_parameters=theta_map.size
        )

        chi_squared_nu = diagnostics.compute_reduced_chi_squared()
        lowess_trend = diagnostics.compute_lowess_trend(frequencies, fraction=args.frac)
        
        sys.stdout.write("Diagnostic evaluation successfully executed.\n")
        sys.stdout.write(f"Extracted MAP Parameter Vector: {theta_map}\n")
        sys.stdout.write(f"Reduced Chi-Squared (Goodness-of-Fit): {chi_squared_nu:.5f}\n")

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