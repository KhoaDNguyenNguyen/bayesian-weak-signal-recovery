#!/usr/bin/env python3
"""
Execution script for rigorous statistical hypothesis testing and Look-Elsewhere Effect (LEE) corrections.

This module ingests the global Bayesian evidence from competing hypotheses (H0 vs. H1),
computes the Bayes Factor, approximates the local significance via asymptotic formulation,
and evaluates the global detection probability applying the Gross-Vitells trials factor.
"""

import argparse
import sys
import pickle
import numpy as np
from pathlib import Path

from bwsr_inference.validation.look_elsewhere import GlobalSignificanceEvaluator


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the significance evaluation sequence.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate Global Statistical Significance via Look-Elsewhere Effect correction.")
    parser.add_argument(
        '--results_h0', 
        type=Path, 
        required=True,
        help="Path to the serialized inference results for the Null Hypothesis (.pkl)."
    )
    parser.add_argument(
        '--results_h1', 
        type=Path, 
        required=True,
        help="Path to the serialized inference results for the Alternative Hypothesis (.pkl)."
    )
    parser.add_argument(
        '--bandwidth', 
        type=float, 
        required=True,
        help="The total continuous frequency bandwidth scanned during the prior integration."
    )
    parser.add_argument(
        '--resolution', 
        type=float, 
        required=True,
        help="The characteristic resolution element (e.g., standard deviation of the target signal)."
    )
    return parser.parse_args()


def extract_log_evidence(results_path: Path) -> float:
    """
    Extract the global Bayesian log-evidence from a serialized Nested Sampling object.

    Parameters
    ----------
    results_path : Path
        The path to the pickled inference output.

    Returns
    -------
    float
        The scalar value corresponding to ln(Z).

    Raises
    ------
    IOError
        If the file cannot be accessed or deserialized.
    """
    try:
        with open(results_path, 'rb') as file_descriptor:
            results = pickle.load(file_descriptor)
        return float(results['logz'][-1])
    except Exception as e:
        raise IOError(f"Failed to extract log-evidence from {results_path}: {e}")


def main() -> None:
    """
    Primary execution sequence for the Bayesian hypothesis testing and LEE correction.
    """
    args = parse_arguments()

    if not args.results_h0.is_file() or not args.results_h1.is_file():
        sys.stderr.write("Error: One or both inference result payloads cannot be located.\n")
        sys.exit(1)

    try:
        logz_h0 = extract_log_evidence(args.results_h0)
        logz_h1 = extract_log_evidence(args.results_h1)

        # Compute the natural logarithm of the Bayes Factor (ln K)
        delta_log_z = logz_h1 - logz_h0

        # Transform Bayes Factor to asymptotic local significance (Wilks' theorem / Gaussian approximation)
        if delta_log_z <= 0.0:
            local_significance = 0.0
            sys.stdout.write("Bayes Factor favors the Null Hypothesis (H0). Asymptotic local significance is exactly zero.\n")
        else:
            local_significance = np.sqrt(2.0 * delta_log_z)

        evaluator = GlobalSignificanceEvaluator(
            search_bandwidth=args.bandwidth,
            signal_resolution=args.resolution
        )

        metrics = evaluator.evaluate(local_significance=local_significance)

        sys.stdout.write("\n--- Statistical Significance & LEE Metrics ---\n")
        sys.stdout.write(f"{'Null Hypothesis Evidence (ln Z0)':<40}: {logz_h0:>15.5f}\n")
        sys.stdout.write(f"{'Alternative Hypothesis Evidence (ln Z1)':<40}: {logz_h1:>15.5f}\n")
        sys.stdout.write(f"{'Log Bayes Factor (Δ ln Z)':<40}: {delta_log_z:>15.5f}\n")
        sys.stdout.write("-" * 58 + "\n")
        sys.stdout.write(f"{'Trials Factor (N_trials)':<40}: {metrics.trials_factor:>15.5f}\n")
        sys.stdout.write(f"{'Local Significance (Z_local)':<40}: {metrics.local_significance:>15.5f} σ\n")
        sys.stdout.write(f"{'Global p-value Bound':<40}: {metrics.global_p_value:>15.5e}\n")
        sys.stdout.write(f"{'Global Significance (Z_global)':<40}: {metrics.global_significance:>15.5f} σ\n")
        sys.stdout.write("-" * 58 + "\n")
        
        if metrics.is_confirmed_detection:
            sys.stdout.write("CONCLUSION: Definitive physical detection confirmed (>= 5.0 σ limit achieved).\n\n")
        else:
            sys.stdout.write("CONCLUSION: Insufficient statistical evidence to falsify the null hypothesis.\n\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()