#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple


def evaluate_exponential_background(
    frequencies: npt.NDArray[np.float64],
    amplitude: float,
    decay_constant: float,
    offset: float
) -> npt.NDArray[np.float64]:
    """
    Compute the deterministic exponential continuum representing non-linear 
    background emission.

    Theoretical Formulation:
        B(f) = A \cdot \exp(-\lambda \cdot f) + C

    Parameters
    ----------
    frequencies : npt.NDArray[np.float64]
        The 1-dimensional array of independent frequency variables.
    amplitude : float
        The baseline scaling factor (A).
    decay_constant : float
        The exponential decay rate (\lambda).
    offset : float
        The constant baseline level (C).

    Returns
    -------
    npt.NDArray[np.float64]
        The evaluated deterministic background array.
    """
    return amplitude * np.exp(-decay_constant * frequencies) + offset


def evaluate_gaussian_signal(
    frequencies: npt.NDArray[np.float64],
    amplitude: float,
    center_frequency: float,
    standard_deviation: float
) -> npt.NDArray[np.float64]:
    """
    Compute the deterministic Gaussian transient representing the latent 
    resonance signature.

    Theoretical Formulation:
        S(f) = A \cdot \exp\left(-\frac{(f - \mu)^2}{2\sigma^2}\right)

    Parameters
    ----------
    frequencies : npt.NDArray[np.float64]
        The 1-dimensional array of independent frequency variables.
    amplitude : float
        The peak magnitude of the transient (A).
    center_frequency : float
        The location parameter or central frequency (\mu).
    standard_deviation : float
        The structural width of the transient (\sigma).

    Returns
    -------
    npt.NDArray[np.float64]
        The evaluated deterministic signal array.

    Raises
    ------
    ValueError
        If the standard deviation assumes a non-positive value.
    """
    if standard_deviation <= 0.0:
        raise ValueError("Standard deviation must be strictly positive.")
    
    variance = standard_deviation ** 2
    return amplitude * np.exp(-0.5 * ((frequencies - center_frequency) ** 2) / variance)


def inject_stochastic_noise(
    deterministic_model: npt.NDArray[np.float64],
    noise_standard_deviation: float,
    random_seed: int = 42
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Inject Additive White Gaussian Noise (AWGN) to synthesize the observable dataset.

    Theoretical Formulation:
        D_i = P(f_i) + \mathcal{N}(0, \sigma_n^2)

    Parameters
    ----------
    deterministic_model : npt.NDArray[np.float64]
        The sum of the deterministic background and signal models.
    noise_standard_deviation : float
        The standard deviation (\sigma_n) of the injected normal distribution.
    random_seed : int, optional
        The seed for the pseudorandom number generator to ensure absolute 
        reproducibility. Default is 42.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        - The synthesized observational data array containing the injected noise.
        - The constant array representing the theoretical variance (\sigma_n^2) 
          for each observational datum.
    """
    if noise_standard_deviation <= 0.0:
        raise ValueError("Noise standard deviation must be strictly positive.")

    generator = np.random.default_rng(seed=random_seed)
    noise_vector = generator.normal(
        loc=0.0, 
        scale=noise_standard_deviation, 
        size=deterministic_model.size
    )
    
    synthesized_data = deterministic_model + noise_vector
    theoretical_variance = np.full_like(deterministic_model, noise_standard_deviation ** 2)
    
    return synthesized_data, theoretical_variance


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configurations for the synthetic generation procedure.

    Returns
    -------
    argparse.Namespace
        The parsed execution parameters.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic frequency spectrum containing a weak latent resonance signature."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/synthetic/mock_spectrum.csv"),
        help="Target output path for the synthesized observational data array."
    )
    parser.add_argument(
        "--f_min", 
        type=float, 
        default=0.0,
        help="Lower boundary of the independent frequency variable."
    )
    parser.add_argument(
        "--f_max", 
        type=float, 
        default=100.0,
        help="Upper boundary of the independent frequency variable."
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=5000,
        help="Total number of discrete sampling bins across the domain."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=12345,
        help="Seed integer for the stochastic noise generator."
    )
    return parser.parse_args()


def main() -> None:
    """
    Primary execution sequence for synthesizing the baseline weak-signal dataset.
    """
    args = parse_arguments()

    if args.f_min >= args.f_max:
        sys.stderr.write("Error: Independent domain constraint violation. Minimum boundary must be strictly less than maximum boundary.\n")
        sys.exit(1)
    if args.resolution <= 0:
        sys.stderr.write("Error: Domain resolution must be a strictly positive integer.\n")
        sys.exit(1)

    try:
        # 1. Initialize Independent Variable Domain
        frequencies = np.linspace(args.f_min, args.f_max, args.resolution, dtype=np.float64)

        # 2. Define Latent Ground Truth Parameters
        # Background Hypothesis: Non-linear Exponential
        bg_amplitude = 25.0
        bg_decay = 0.05
        bg_offset = 5.0
        
        # Signal Hypothesis: Weak Gaussian Transient
        sig_amplitude = 1.2
        sig_center = 42.5
        sig_width = 0.8
        
        # Stochastic Assumption: Additive White Gaussian Noise
        noise_std = 3.5  # Substantial variance to ensure low Signal-to-Noise Ratio (SNR)

        # 3. Construct Deterministic Physical Models
        background_model = evaluate_exponential_background(
            frequencies, bg_amplitude, bg_decay, bg_offset
        )
        signal_model = evaluate_gaussian_signal(
            frequencies, sig_amplitude, sig_center, sig_width
        )
        total_deterministic_model = background_model + signal_model

        # 4. Inject Stochastic Processes
        observational_data, variance_array = inject_stochastic_noise(
            deterministic_model=total_deterministic_model,
            noise_standard_deviation=noise_std,
            random_seed=args.seed
        )

        # 5. Serialize Output Matrix
        # Integration requirement: Pipeline expects Frequency, Power Spectral Density, and Variance
        dataset = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Power_Spectral_Density': observational_data,
            'Variance': variance_array
        })

        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.output, index=False)

        # Output explicit parameter confirmation to standard output
        sys.stdout.write("Synthetic data generation successfully executed.\n")
        sys.stdout.write("--- Latent Ground Truth Parameters ---\n")
        sys.stdout.write(f"Background Model: Amplitude={bg_amplitude}, Decay={bg_decay}, Offset={bg_offset}\n")
        sys.stdout.write(f"Signal Model: Amplitude={sig_amplitude}, Center={sig_center}, Width={sig_width}\n")
        sys.stdout.write(f"Noise Statistics: Standard Deviation={noise_std}, Global Variance={noise_std**2}\n")
        sys.stdout.write(f"Target Output Path: {args.output}\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to operational error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()