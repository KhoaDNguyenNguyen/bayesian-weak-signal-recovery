#!/usr/bin/env python3
"""
High-Performance Computing (HPC) Orchestrator for Trans-dimensional RJ-MCMC.

This module deploys an ensemble of independent Markov Chains across multiple 
hardware threads, leveraging CPU concurrency to dramatically amplify the 
effective sample size and numerical precision of the Empirical Bayes Factor.
"""

import argparse
import sys
import os
import json
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, Any, Tuple

from bwsr_inference.forward_model import MODEL_ZOO
from bwsr_inference.sampling.likelihood import DiagonalGaussianLikelihood
from bwsr_inference.sampling.rjmcmc import ReversibleJumpMCMC


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
    Parse command-line configurations for the ensemble RJ-MCMC execution.
    """
    parser = argparse.ArgumentParser(description="HPC Ensemble Trans-dimensional RJ-MCMC Pipeline.")
    parser.add_argument(
        '--input', 
        type=Path, 
        required=True, 
        help="Path to the preprocessed Layer 1 observational CSV dataset."
    )
    parser.add_argument(
        '--priors', 
        type=Path, 
        default=Path("config/inference_priors.json"), 
        help="Path to the JSON schema defining the parameter topologies."
    )
    parser.add_argument(
        '--iter', 
        type=int, 
        default=100000, 
        help="Number of Markov Chain iterations per independent thread."
    )
    parser.add_argument(
        '--burnin', 
        type=int, 
        default=20000, 
        help="Number of iterations discarded for thermalization per thread."
    )
    parser.add_argument(
        '--threads', 
        type=int, 
        default=_determine_optimal_thread_count(), 
        help="Number of parallel Markov Chains to deploy (defaults to max CPU cores)."
    )
    return parser.parse_args()


def load_model_config(config_path: Path, model_id: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Extract geometric boundaries and scaling configurations from the JSON schema.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    if model_id not in schema:
        raise KeyError(f"Configuration '{model_id}' missing from schema.")

    bounds = np.array(schema[model_id]['bounds'], dtype=np.float64)
    flags = np.array(schema[model_id]['log_scale_flags'], dtype=np.bool_)
    return bounds, flags


def _rjmcmc_worker(
    worker_id: int, 
    seed: int, 
    input_path: Path, 
    priors_path: Path, 
    n_iterations: int, 
    burn_in: int
) -> Dict[str, Any]:
    """
    Isolated execution sequence for a single Markov Chain. This function is 
    designed to be strictly picklable for the multiprocessing pool.
    """
    try:
        # Independent data parsing to prevent memory contention
        data_matrix = np.genfromtxt(input_path, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        observational_data = data_matrix[:, 1]
        noise_variance = data_matrix[:, 2]

        bounds_h0, flags_h0 = load_model_config(priors_path, 'h0')
        bounds_h1, flags_h1 = load_model_config(priors_path, 'h1_gauss')

        forward_model_h0 = MODEL_ZOO['h0']
        forward_model_h1 = MODEL_ZOO['h1_gauss']

        likelihood_h0 = DiagonalGaussianLikelihood(frequencies, observational_data, noise_variance, forward_model_h0)
        likelihood_h1 = DiagonalGaussianLikelihood(frequencies, observational_data, noise_variance, forward_model_h1)

        rjmcmc_engine = ReversibleJumpMCMC(
            likelihood_h0=likelihood_h0,
            likelihood_h1=likelihood_h1,
            prior_bounds_h0=bounds_h0,
            prior_bounds_h1=bounds_h1,
            log_scale_flags_h1=flags_h1,
            random_seed=seed
        )

        results = rjmcmc_engine.execute(n_iterations=n_iterations, burn_in=burn_in)
        
        model_indicator = results["model_indicator"]
        n_h0 = int(np.sum(model_indicator == 0))
        n_h1 = int(np.sum(model_indicator == 1))
        
        return {
            "worker_id": worker_id,
            "n_h0": n_h0,
            "n_h1": n_h1,
            "acceptance_rate": results["jump_acceptance_rate"]
        }
    except Exception as e:
        return {"worker_id": worker_id, "error": str(e)}


def main() -> None:
    """
    Primary orchestrator sequence for the HPC Trans-dimensional exploration.
    """
    args = parse_arguments()

    if not args.input.is_file():
        sys.stderr.write(f"Error: Observational dataset '{args.input}' not found.\n")
        sys.exit(1)

    try:
        sys.stdout.write("--- HPC Trans-dimensional RJ-MCMC Ensemble Engine ---\n")
        sys.stdout.write(f"Allocated Hardware Threads : {args.threads}\n")
        sys.stdout.write(f"Iterations per Thread      : {args.iter} (Burn-in: {args.burnin})\n")
        sys.stdout.write(f"Total Ensemble Iterations  : {args.threads * args.iter}\n\n")
        sys.stdout.write("Deploying independent Markov Chains across CPU cores...\n")

        # Formulate argument matrix for parallel map execution
        base_seed = 42
        worker_args = [
            (i, base_seed + i * 137, args.input, args.priors, args.iter, args.burnin) 
            for i in range(args.threads)
        ]

        # Execute parallel ensemble integration
        with mp.Pool(processes=args.threads) as pool:
            ensemble_results = pool.starmap(_rjmcmc_worker, worker_args)

        # Global Aggregation
        global_n_h0 = 0
        global_n_h1 = 0
        acceptance_rates = []

        for res in ensemble_results:
            if "error" in res:
                sys.stderr.write(f"Worker {res['worker_id']} failed: {res['error']}\n")
                continue
            
            global_n_h0 += res["n_h0"]
            global_n_h1 += res["n_h1"]
            acceptance_rates.append(res["acceptance_rate"])

        if len(acceptance_rates) == 0:
            raise RuntimeError("All computational threads aborted unexpectedly.")

        total_valid_samples = global_n_h0 + global_n_h1
        if total_valid_samples == 0:
            raise ValueError("Burn-in period exceeds or equals the total iterations. No valid samples were collected.")
        
        p_h0 = global_n_h0 / total_valid_samples
        p_h1 = global_n_h1 / total_valid_samples
        mean_acceptance = np.mean(acceptance_rates)

        empirical_bayes_factor = global_n_h1 / max(1, global_n_h0)
        log_k = np.log(empirical_bayes_factor) if empirical_bayes_factor > 0 else -np.inf

        sys.stdout.write("\n--- Global Stationarity & Ensemble Metrics ---\n")
        sys.stdout.write(f"Aggregate Valid Samples           : {total_valid_samples}\n")
        sys.stdout.write(f"Mean Trans-dimensional Acceptance : {mean_acceptance * 100:.2f} %\n")
        sys.stdout.write(f"H0 Global Visitation Freq (P|D)   : {p_h0 * 100:.3f} %\n")
        sys.stdout.write(f"H1 Global Visitation Freq (P|D)   : {p_h1 * 100:.3f} %\n")
        sys.stdout.write(f"Empirical Log Bayes Factor (ln K) : {log_k:.5f}\n")
        
        if log_k > 3.0:
            sys.stdout.write("\nCONCLUSION: The ensemble conclusively favors the Trans-dimensional Signal Space (H1).\n")
        elif log_k < -3.0:
            sys.stdout.write("\nCONCLUSION: The ensemble conclusively defaults to the Baseline Noise Space (H0).\n")
        else:
            sys.stdout.write("\nCONCLUSION: Markov Chains exhibit continuous dimensional oscillations. Insufficient physical evidence.\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational failure: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    # Enforce forkserver semantics to guarantee robust isolation of C-extension 
    # threading states and unconditionally prevent pre-forking process deadlocks.
    mp.set_start_method('forkserver', force=True)
    main()