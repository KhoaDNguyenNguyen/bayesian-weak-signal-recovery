#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from pathlib import Path

# Corrected namespace from ssr_inference to bwsr_inference
from bwsr_inference.forward_model.signal import GaussianSignal
from bwsr_inference.sampling.likelihood import PriorTransform, DiagonalGaussianLikelihood
from bwsr_inference.sampling.nested_sampler import NestedInferenceEngine


def parse_arguments() -> argparse.Namespace:
    r"""
    Parse command-line arguments for High-Performance Computing batch execution.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute the Bayesian Inference pipeline via Nested Sampling."
    )
    parser.add_argument(
        '--input', 
        type=Path, 
        required=True,
        help="Path to the Layer 1 observational data CSV (e.g., data/processed/baseline_spectrum.csv)."
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path("data/processed/inference_results.pkl"),
        help="Target output path for the serialized binary results."
    )
    parser.add_argument(
        '--nlive', 
        type=int, 
        default=1000,
        help="Number of live points for the Nested Sampling algorithm."
    )
    parser.add_argument(
        '--tol', 
        type=float, 
        default=0.1,
        help="Convergence tolerance on the log-evidence (dlogz)."
    )
    return parser.parse_args()


def main() -> None:
    r"""
    Main execution entry point for the Bayesian inference operation.
    """
    args = parse_arguments()

    if not args.input.is_file():
        sys.stderr.write(f"Error: Input data file {args.input} does not exist.\n")
        sys.exit(1)

    try:
        data_matrix = np.genfromtxt(args.input, delimiter=',', skip_header=1)
        if data_matrix.shape[1] < 3:
            raise ValueError("Input data must contain at least 3 columns: Frequency, Power, Variance.")
            
        frequencies = data_matrix[:, 0]
        observational_data = data_matrix[:, 1]
        noise_variance = data_matrix[:, 2]
    except Exception as e:
        sys.stderr.write(f"Error during data ingestion: {e}\n")
        sys.exit(1)

    prior_bounds = np.array([
        [1e-6, 1e-1],
        [np.min(frequencies), np.max(frequencies)],
        [1e-3, 10.0]
    ])
    
    log_scale_flags = np.array([True, False, True], dtype=bool)

    try:
        prior_transform = PriorTransform(bounds=prior_bounds, log_flags=log_scale_flags)
        forward_model = GaussianSignal()
        
        likelihood_formulation = DiagonalGaussianLikelihood(
            frequencies=frequencies,
            data=observational_data,
            noise_variance=noise_variance,
            forward_model=forward_model
        )
        
        inference_engine = NestedInferenceEngine(
            log_likelihood=likelihood_formulation,
            prior_transform=prior_transform,
            n_dim=prior_bounds.shape[0],
            n_live_points=args.nlive
        )
    except Exception as e:
        sys.stderr.write(f"Error during component initialization: {e}\n")
        sys.exit(1)

    sys.stdout.write("Initiating nested sampling sequence...\n")
    try:
        results = inference_engine.execute(dlogz=args.tol)
        
        sys.stdout.write("Inference execution successfully concluded.\n")
        sys.stdout.write(f"Global Log-Evidence (ln Z): {results['log_evidence']:.5f} +/- {results['log_evidence_err']:.5f}\n")
        sys.stdout.write(f"Effective Sample Size: {len(results['samples'])}\n")
        
        inference_engine.serialize_results(args.output)
        sys.stdout.write(f"Results serialized to {args.output}\n")
        
    except Exception as e:
        sys.stderr.write(f"Error during inference execution or serialization: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()