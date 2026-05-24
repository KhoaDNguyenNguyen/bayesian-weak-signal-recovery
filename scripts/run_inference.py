#!/usr/bin/env python3
import argparse
import sys
import os
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from pathlib import Path

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal
from bwsr_inference.forward_model.background import ExponentialBackground
from bwsr_inference.sampling.likelihood import PriorTransform, DiagonalGaussianLikelihood
from bwsr_inference.sampling.nested_sampler import NestedInferenceEngine


def _determine_optimal_thread_count() -> int:
    r"""
    Dynamically ascertain the maximum optimal number of processing threads 
    available to the current execution context.

    This function prioritizes POSIX CPU affinity constraints to prevent 
    over-subscription in High-Performance Computing (HPC) environments 
    governed by resource schedulers (e.g., SLURM, PBS) or containerized limits.

    Returns
    -------
    int
        The strictly positive integer representing the available processing threads.
    """
    try:
        # Strictly valid on POSIX architectures; acquires process-specific CPU mask.
        optimal_count = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback mechanism for non-POSIX architectures.
        optimal_count = os.cpu_count()
        if optimal_count is None:
            optimal_count = 1
            
    return optimal_count


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
            
        Raises
        ------
        ValueError
            If the parameter vector does not contain exactly 6 elements.
        """
        if theta.size != 6:
            raise ValueError(f"CompositeForwardModel requires exactly 6 parameters. Received {theta.size}.")

        return self._background_model(frequencies, theta[0:3]) + \
               self._signal_model(frequencies, theta[3:6])


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
        help="Path to the Layer 1 observational data CSV."
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
    parser.add_argument(
        '--threads', 
        type=int, 
        default=_determine_optimal_thread_count(),
        help="Number of concurrent multiprocessing threads. Defaults to maximum available hardware concurrency."
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

    # Prior bounds accommodating the 6-dimensional composite state space
    prior_bounds = np.array([
        [1.0, 100.0],                                  # BG Amplitude
        [1e-3, 1.0],                                   # BG Decay Constant
        [0.0, 50.0],                                   # BG Offset
        [1e-2, 10.0],                                  # Signal Amplitude
        [40.0, 45.0],                                  # TARGETED: Signal Center Frequency
        [1e-2, 10.0]                                   # Signal Width
    ], dtype=np.float64)
    
    log_scale_flags = np.array([True, True, False, True, False, True], dtype=np.bool_)

    try:
        prior_transform = PriorTransform(bounds=prior_bounds, log_flags=log_scale_flags)
        forward_model = CompositeForwardModel()
        
        likelihood_formulation = DiagonalGaussianLikelihood(
            frequencies=frequencies,
            data=observational_data,
            noise_variance=noise_variance,
            forward_model=forward_model
        )
    except Exception as e:
        sys.stderr.write(f"Error during component initialization: {e}\n")
        sys.exit(1)

    sys.stdout.write(f"Initiating nested sampling sequence via {args.threads} parallel hardware threads...\n")
    
    try:
        # Establish the symmetrical multiprocessing pool context
        with mp.Pool(processes=args.threads) as process_pool:
            inference_engine = NestedInferenceEngine(
                log_likelihood=likelihood_formulation,
                prior_transform=prior_transform,
                n_dim=prior_bounds.shape[0],
                n_live_points=args.nlive,
                sampling_method='rwalk',  # Random Walk MCMC enforcement to navigate posterior degeneracies
                pool=process_pool,
                queue_size=args.threads
            )
            
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
    # Enforce POSIX memory mapping optimization to mitigate data duplication across processes
    mp.set_start_method('fork', force=True)
    main()