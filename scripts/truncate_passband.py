#!/usr/bin/env python3
r"""
Deterministic frequency domain truncation module.

This operational script isolates the linear passband of an empirical 
Power Spectral Density (PSD) dataset, discarding transition bands and 
stopbands to prevent inverse-variance weighting explosions and structural 
model misspecifications during Bayesian inference.

Theoretical Formulation:
    Given an observational set D = {(f_i, P_i, \sigma_i^2)}, the truncation 
    operator T extracts the subset D' such that:
        D' = { (f_i, P_i, \sigma_i^2) \in D \mid f_{min} \le f_i \le f_{max} }
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the passband truncation procedure.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments specifying I/O paths and domain constraints.
    """
    parser = argparse.ArgumentParser(
        description="Extract a localized frequency passband from Layer 1 observational data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source Layer 1 observational dataset (.csv)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Target output path for the truncated observational dataset (.csv)."
    )
    parser.add_argument(
        "--f_min",
        type=float,
        required=True,
        help="Lower bound of the independent frequency domain (f_min)."
    )
    parser.add_argument(
        "--f_max",
        type=float,
        required=True,
        help="Upper bound of the independent frequency domain (f_max)."
    )
    return parser.parse_args()


def execute_truncation(
    input_path: Path, 
    output_path: Path, 
    f_min: float, 
    f_max: float
) -> None:
    """
    Execute the deterministic subset extraction based on frequency constraints.

    Parameters
    ----------
    input_path : Path
        The file system path to the input dataset.
    output_path : Path
        The file system path where the truncated dataset will be serialized.
    f_min : float
        The inclusive lower frequency boundary.
    f_max : float
        The inclusive upper frequency boundary.

    Raises
    ------
    FileNotFoundError
        If the specified input dataset cannot be located.
    ValueError
        If the frequency boundaries are mathematically invalid or if the 
        resulting dataset subset is empty.
    RuntimeError
        If data serialization encounters an I/O exception.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Source observational dataset '{input_path}' not found.")
    
    if f_min >= f_max:
        raise ValueError(
            f"Domain constraint violation: f_min ({f_min}) must be strictly "
            f"less than f_max ({f_max})."
        )

    try:
        dataset = pd.read_csv(input_path)
    except Exception as e:
        raise IOError(f"Failure during dataset ingestion: {e}")

    if 'Frequency_Hz' not in dataset.columns:
        raise ValueError("Dataset schema violation: Missing 'Frequency_Hz' column.")

    # Apply strictly inclusive boundary conditions
    mask = (dataset['Frequency_Hz'] >= f_min) & (dataset['Frequency_Hz'] <= f_max)
    truncated_dataset = dataset.loc[mask].copy()

    n_samples_retained = len(truncated_dataset)
    if n_samples_retained == 0:
        raise ValueError(
            f"The specified domain [{f_min}, {f_max}] yielded an empty dataset. "
            "Ensure the bounds reside within the empirical frequency range."
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        truncated_dataset.to_csv(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failure during dataset serialization: {e}")

    sys.stdout.write("Passband truncation successfully executed.\n")
    sys.stdout.write(f"Domain Constraints : [{f_min} Hz, {f_max} Hz]\n")
    sys.stdout.write(f"Samples Retained   : {n_samples_retained} / {len(dataset)}\n")
    sys.stdout.write(f"Serialized Payload : {output_path}\n")


def main() -> None:
    """
    Primary execution sequence for the passband isolation module.
    """
    args = parse_arguments()

    try:
        execute_truncation(
            input_path=args.input,
            output_path=args.output,
            f_min=args.f_min,
            f_max=args.f_max
        )
    except Exception as e:
        sys.stderr.write(f"Execution aborted due to operational error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()