#!/usr/bin/env python3
import argparse
import sys
import os
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from pathlib import Path

from bwsr_inference.forward_model.signal import AbstractForwardModel, GaussianSignal
from bwsr_inference.forward_model.background import PolynomialBackground
from bwsr_inference.sampling.likelihood import PriorTransform, DiagonalGaussianLikelihood
from bwsr_inference.sampling.nested_sampler import NestedInferenceEngine

def _determine_optimal_thread_count() -> int:
    try:
        optimal_count = len(os.sched_getaffinity(0))
    except AttributeError:
        optimal_count = os.cpu_count()
        if optimal_count is None:
            optimal_count = 1
    return optimal_count

class CompositeForwardModel(AbstractForwardModel):
    def __init__(self) -> None:
        self._background_model = PolynomialBackground()
        self._signal_model = GaussianSignal()

    def __call__(self, frequencies: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if theta.size != 4:
            raise ValueError(f"H1 Model requires exactly 4 parameters.")
        return self._background_model(frequencies, theta[0:1]) + \
               self._signal_model(frequencies, theta[1:4])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian Inference Pipeline")
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, choices=['h0', 'h1'], required=True, help="Hypothesis to test: h0 (Noise) or h1 (Signal)")
    parser.add_argument('--nlive', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=_determine_optimal_thread_count())
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    if not args.input.is_file():
        sys.exit(1)

    data_matrix = np.genfromtxt(args.input, delimiter=',', skip_header=1)
    frequencies = data_matrix[:, 0]
    observational_data = data_matrix[:, 1]
    noise_variance = data_matrix[:, 2]

    # Dynamically configure priors and models based on hypothesis testing
    if args.model == 'h0':
        prior_bounds = np.array([[1e-6, 1e-4]], dtype=np.float64) # Only AWGN Floor
        log_scale_flags = np.array([True], dtype=np.bool_)
        forward_model = PolynomialBackground()
    else:
        # Restored Log-Uniform Prior for stable weak signal extraction
        prior_bounds = np.array([
            [1e-6, 1e-4],                                  # AWGN Floor
            [1e-8, 1e-4],                                  # Signal Amplitude
            [9500.0, 10500.0],                             # Signal Center
            [1.0, 500.0]                                   # Signal Width
        ], dtype=np.float64)
        log_scale_flags = np.array([True, True, False, True], dtype=np.bool_)
        forward_model = CompositeForwardModel()

    prior_transform = PriorTransform(bounds=prior_bounds, log_flags=log_scale_flags)
    
    likelihood_formulation = DiagonalGaussianLikelihood(
        frequencies=frequencies, data=observational_data,
        noise_variance=noise_variance, forward_model=forward_model
    )

    sys.stdout.write(f"Executing Hypothesis {args.model.upper()} via {args.threads} threads...\n")
    
    with mp.Pool(processes=args.threads) as process_pool:
        inference_engine = NestedInferenceEngine(
            log_likelihood=likelihood_formulation, prior_transform=prior_transform,
            n_dim=prior_bounds.shape[0], n_live_points=args.nlive,
            sampling_method='rwalk', pool=process_pool, queue_size=args.threads
        )
        results = inference_engine.execute(dlogz=args.tol)
    
    sys.stdout.write(f"Global Log-Evidence (ln Z): {results['log_evidence']:.5f} +/- {results['log_evidence_err']:.5f}\n")
    inference_engine.serialize_results(args.output)

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    main()