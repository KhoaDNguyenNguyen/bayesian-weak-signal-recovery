#!/usr/bin/env python3
"""
High-Performance Computing (HPC) Orchestrator for Generalized Trans-dimensional RJ-MCMC.

This module deploys an ensemble of independent Markov Chains across multiple 
hardware threads, dynamically parsing topological configurations to evaluate 
an arbitrary dimensional hypothesis space.
"""

import argparse
import sys
import os
import json
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, Any, List

from bwsr_inference.forward_model import MODEL_ZOO
from bwsr_inference.sampling.likelihood import DiagonalGaussianLikelihood
from bwsr_inference.sampling.rjmcmc import GeneralizedReversibleJumpMCMC, ModelHypothesis


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
        help="Number of parallel Markov Chains to deploy."
    )
    return parser.parse_args()


def _rjmcmc_worker(
    worker_id: int, 
    seed: int, 
    input_path: Path, 
    priors_path: Path, 
    n_iterations: int, 
    burn_in: int
) -> Dict[str, Any]:
    """
    Isolated execution sequence for a single Markov Chain utilizing dynamic schema parsing.
    """
    try:
        data_matrix = np.genfromtxt(input_path, delimiter=',', skip_header=1)
        frequencies = data_matrix[:, 0]
        observational_data = data_matrix[:, 1]
        noise_variance = data_matrix[:, 2]

        with open(priors_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        hypotheses: List[ModelHypothesis] = []
        for model_id, config in schema.items():
            if model_id not in MODEL_ZOO:
                continue
                
            bounds = np.array(config['bounds'], dtype=np.float64)
            flags = np.array(config['log_scale_flags'], dtype=np.bool_)
            
            forward_model = MODEL_ZOO[model_id]
            likelihood = DiagonalGaussianLikelihood(
                frequencies, observational_data, noise_variance, forward_model
            )
            
            hypotheses.append(ModelHypothesis(
                model_id=model_id,
                log_likelihood=likelihood,
                prior_bounds=bounds,
                log_scale_flags=flags
            ))

        if len(hypotheses) < 2:
            raise ValueError("Insufficient architectural definitions. RJ-MCMC requires >= 2 valid models.")

        rjmcmc_engine = GeneralizedReversibleJumpMCMC(
            hypotheses=hypotheses,
            random_seed=seed
        )

        results = rjmcmc_engine.execute(n_iterations=n_iterations, burn_in=burn_in)
        
        model_indicator = results["model_indicator"]
        visitation_counts: Dict[str, int] = {}
        
        for idx, hyp in enumerate(hypotheses):
            visitation_counts[hyp.model_id] = int(np.sum(model_indicator == idx))
            
        return {
            "worker_id": worker_id,
            "visitation_counts": visitation_counts,
            "acceptance_rate": results["jump_acceptance_rate"]
        }
    except Exception as e:
        return {"worker_id": worker_id, "error": str(e)}


def main() -> None:
    """
    Primary orchestrator sequence for the generalized HPC Trans-dimensional exploration.
    """
    args = parse_arguments()

    if not args.input.is_file():
        sys.stderr.write(f"Error: Observational dataset '{args.input}' not found.\n")
        sys.exit(1)

    try:
        sys.stdout.write("--- HPC Generalized Trans-dimensional RJ-MCMC Ensemble Engine ---\n")
        sys.stdout.write(f"Allocated Hardware Threads : {args.threads}\n")
        sys.stdout.write(f"Iterations per Thread      : {args.iter} (Burn-in: {args.burnin})\n")
        sys.stdout.write(f"Total Ensemble Iterations  : {args.threads * args.iter}\n\n")
        sys.stdout.write("Deploying independent Markov Chains across CPU cores...\n")

        base_seed = 42
        worker_args = [
            (i, base_seed + i * 137, args.input, args.priors, args.iter, args.burnin) 
            for i in range(args.threads)
        ]

        with mp.Pool(processes=args.threads) as pool:
            ensemble_results = pool.starmap(_rjmcmc_worker, worker_args)

        global_counts: Dict[str, int] = {}
        acceptance_rates: List[float] = []

        for res in ensemble_results:
            if "error" in res:
                sys.stderr.write(f"Worker {res['worker_id']} failed: {res['error']}\n")
                continue
            
            for m_id, count in res["visitation_counts"].items():
                global_counts[m_id] = global_counts.get(m_id, 0) + count
            
            acceptance_rates.append(res["acceptance_rate"])

        if not acceptance_rates:
            raise RuntimeError("All computational threads aborted unexpectedly.")

        total_valid_samples = sum(global_counts.values())
        if total_valid_samples == 0:
            raise ValueError("Burn-in period exceeds or equals the total iterations.")
        
        mean_acceptance = np.mean(acceptance_rates)

        sys.stdout.write("\n--- Global Stationarity & Ensemble Metrics ---\n")
        sys.stdout.write(f"Aggregate Valid Samples           : {total_valid_samples}\n")
        sys.stdout.write(f"Mean Trans-dimensional Acceptance : {mean_acceptance * 100:.2f} %\n")
        sys.stdout.write("-" * 58 + "\n")
        
        for model_id, count in sorted(global_counts.items()):
            probability = count / total_valid_samples
            sys.stdout.write(f"{model_id.upper():<33} : {probability * 100:>8.3f} %\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational failure: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    main()