#!/usr/bin/env python3
"""
Execution script for Bayesian inference and Nested Sampling integration.

This module orchestrates the ingestion of empirical spectral datasets, dynamically
constructs the likelihood formulations based on an external configuration schema,
and executes multi-threaded probability space exploration.
"""

import argparse
import sys
import os
import json
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple

from bwsr_inference.forward_model import MODEL_ZOO
from bwsr_inference.sampling.likelihood import PriorTransform, DiagonalGaussianLikelihood
from bwsr_inference.sampling.nested_sampler import NestedInferenceEngine


def _determine_optimal_thread_count() -> int:
    """
    Determine the optimal number of execution threads based on CPU affinity.
    """
    try:
        optimal_count = len(os.sched_getaffinity(0))
    except AttributeError:
        optimal_count = os.cpu_count()
        if optimal_count is None:
            optimal_count = 1
    return optimal_count


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the Bayesian inference pipeline.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Bayesian Inference Pipeline Execution Engine.")
    parser.add_argument(
        '--input', 
        type=Path, 
        required=True, 
        help="Path to the preprocessed Layer 1 observational CSV dataset."
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True, 
        help="Target output path for the serialized inference payload (.pkl)."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=list(MODEL_ZOO.keys()), 
        required=True, 
        help="Hypothesis specification to dictate forward model compilation."
    )
    parser.add_argument(
        '--priors', 
        type=Path, 
        default=Path("config/inference_priors.json"), 
        help="Path to the JSON schema defining the prior bounds and transformations."
    )
    parser.add_argument(
        '--nlive', 
        type=int, 
        default=1000, 
        help="Number of active live points for the Nested Sampling algorithm."
    )
    parser.add_argument(
        '--tol', 
        type=float, 
        default=0.1, 
        help="Log-evidence tolerance threshold for termination criteria."
    )
    parser.add_argument(
        '--threads', 
        type=int, 
        default=_determine_optimal_thread_count(), 
        help="Total number of threads allocated for parallel likelihood evaluation."
    )
    return parser.parse_args()


def load_prior_configuration(config_path: Path, model_id: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Ingest and strictly validate the prior boundaries and scaling flags from 
    the external JSON configuration matrix.

    Parameters
    ----------
    config_path : Path
        The file system path to the JSON configuration.
    model_id : str
        The dictionary key corresponding to the target hypothesis.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]
        A tuple containing the 2D array of boundaries and the 1D boolean array 
        dictating logarithmic scaling.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    KeyError
        If the specified model_id or required schema keys are missing.
    ValueError
        If dimensionalities between boundaries and flags mismatch.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Prior configuration schema cannot be located at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    if model_id not in schema:
        raise KeyError(f"Configuration for hypothesis '{model_id}' is missing from the JSON schema.")

    model_config = schema[model_id]

    try:
        bounds_list = model_config['bounds']
        log_flags_list = model_config['log_scale_flags']
    except KeyError as e:
        raise KeyError(f"Malformed configuration schema for '{model_id}'. Missing key: {e}")

    prior_bounds = np.array(bounds_list, dtype=np.float64)
    log_scale_flags = np.array(log_flags_list, dtype=np.bool_)

    if prior_bounds.ndim != 2 or prior_bounds.shape[1] != 2:
        raise ValueError("Prior bounds must form a 2-dimensional matrix of shape (D, 2).")
    
    if prior_bounds.shape[0] != log_scale_flags.shape[0]:
        raise ValueError(
            f"Dimensionality mismatch: Defined bounds (D={prior_bounds.shape[0]}) "
            f"do not align with log_scale_flags (D={log_scale_flags.shape[0]})."
        )

    return prior_bounds, log_scale_flags


def main() -> None:
    """
    Primary execution sequence for Bayesian hypothesis testing and integration.
    """
    args = parse_arguments()

    if not args.input.is_file():
        sys.stderr.write(f"Error: Observational data {args.input} not found. Ensure preprocessing phase is complete.\n")
        sys.exit(1)

    try:
        # 1. Observational Data Ingestion
        data_matrix = np.genfromtxt(args.input, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        observational_data = data_matrix[:, 1]
        noise_variance = data_matrix[:, 2]

        # 2. Dynamic Prior & Forward Model Compilation
        prior_bounds, log_scale_flags = load_prior_configuration(args.priors, args.model)
        forward_model = MODEL_ZOO[args.model]
        
        prior_transform = PriorTransform(bounds=prior_bounds, log_flags=log_scale_flags)
        
        likelihood_formulation = DiagonalGaussianLikelihood(
            frequencies=frequencies, 
            data=observational_data,
            noise_variance=noise_variance, 
            forward_model=forward_model
        )

        sys.stdout.write(f"Evaluating Hypothesis [{args.model.upper()}] with {prior_bounds.shape[0]} degrees of freedom.\n")
        sys.stdout.write(f"Prior configuration ingested from: {args.priors}\n")
        sys.stdout.write(f"Allocating {args.threads} hardware threads for parallel sampling...\n\n")
        
        # 3. Probability Space Integration via Nested Sampling
        with mp.Pool(processes=args.threads) as process_pool:
            inference_engine = NestedInferenceEngine(
                log_likelihood=likelihood_formulation, 
                prior_transform=prior_transform,
                n_dim=prior_bounds.shape[0], 
                n_live_points=args.nlive,
                sampling_method='rwalk', 
                pool=process_pool, 
                queue_size=args.threads
            )
            results = inference_engine.execute(dlogz=args.tol)
        
        # 4. Payload Serialization
        sys.stdout.write(f"\nGlobal Log-Evidence (ln Z): {results['log_evidence']:.5f} +/- {results['log_evidence_err']:.5f}\n")
        inference_engine.serialize_results(args.output)
        sys.stdout.write(f"Posterior distribution and evidence payload serialized to: {args.output}\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational or architectural error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    # Enforce forkserver semantics to guarantee robust isolation of C-extension 
    # threading states (e.g., OpenBLAS) and unconditionally prevent process deadlocks.
    mp.set_start_method('forkserver', force=True)
    main()