#!/usr/bin/env python3
"""
Execution script for asymptotic limit evaluation and Cramér-Rao Lower Bound (CRLB) analysis.

This module computes the Fisher Information Matrix (FIM) from the Maximum A Posteriori (MAP)
parameter estimates and contrasts the theoretical variance bounds against the empirical
posterior variance extracted via Nested Sampling.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import numpy.typing as npt

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal
from bwsr_inference.forward_model.background import PolynomialBackground
from bwsr_inference.validation.crlb import evaluate_asymptotic_limits


class CompositeForwardModel(AbstractForwardModel):
    """
    Constructs the exact deterministic mapping function evaluated during the inference phase.
    """
    def __init__(self) -> None:
        self._background_model = PolynomialBackground()
        self._signal_model = GaussianSignal()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if theta.size != 4:
            raise ValueError(f"Composite model expects exactly 4 parameters, received {theta.size}.")
        return self._background_model(frequencies, theta[0:1]) + self._signal_model(frequencies, theta[1:4])


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the CRLB evaluation pipeline.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate Cramér-Rao Lower Bounds and Fisher Information.")
    parser.add_argument(
        '--data', 
        type=Path, 
        required=True,
        help="Path to the Layer 1 observational CSV data."
    )
    parser.add_argument(
        '--results', 
        type=Path, 
        required=True,
        help="Path to the serialized Nested Sampling inference results (.pkl)."
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help="Target output path for the CRLB vector diagnostic graphic (.pdf)."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['h0', 'h1'], 
        required=True,
        help="Hypothesis classification to dictate the forward model construction."
    )
    return parser.parse_args()


def main() -> None:
    """
    Primary execution sequence for the Fisher Information Matrix computation.
    """
    args = parse_arguments()

    if not args.data.is_file():
        sys.stderr.write(f"Error: Observational data file {args.data} cannot be located.\n")
        sys.exit(1)
    if not args.results.is_file():
        sys.stderr.write(f"Error: Inference results file {args.results} cannot be located.\n")
        sys.exit(1)

    if args.model == 'h0':
        forward_model = PolynomialBackground()
        parameter_labels = ["AWGN_Floor"]
    else:
        forward_model = CompositeForwardModel()
        parameter_labels = ["AWGN_Floor", "Sig_Amp", "Sig_Center", "Sig_Width"]

    try:
        sys.stdout.write("Executing Fisher Information Matrix computations...\n")
        
        evaluation_metrics = evaluate_asymptotic_limits(
            inference_results_path=args.results,
            data_path=args.data,
            forward_model=forward_model,
            output_plot_path=args.output
        )

        sys.stdout.write("\n--- Asymptotic Efficiency Metrics ---\n")
        sys.stdout.write(f"{'Parameter':<15} | {'MAP Estimate':<15} | {'CRLB (Theoretical)':<20} | {'Posterior Variance':<20}\n")
        sys.stdout.write("-" * 80 + "\n")

        theta_map = evaluation_metrics["theta_map"]
        crlb = evaluation_metrics["cramer_rao_lower_bound"]
        empirical_variance = evaluation_metrics["empirical_variance"]

        for idx, label in enumerate(parameter_labels):
            sys.stdout.write(
                f"{label:<15} | {theta_map[idx]:>15.8e} | {crlb[idx]:>20.8e} | {empirical_variance[idx]:>20.8e}\n"
            )

        sys.stdout.write(f"\nCRLB diagnostic graphic successfully exported to: {args.output}\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()