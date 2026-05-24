#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal
from pathlib import Path
from typing import Tuple


def parse_arguments() -> argparse.Namespace:
    r"""
    Parse command-line configurations for the Software-Defined Radio (SDR) 
    preprocessing pipeline.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transform raw complex SDR time-series data into statistical Power Spectral Density distributions."
    )
    parser.add_argument(
        '--input', 
        type=Path, 
        required=True,
        help="Path to the raw binary SDR data file (.dat)."
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help="Target output path for the Layer 1 observational CSV."
    )
    parser.add_argument(
        '--fs', 
        type=float, 
        default=32000.0,
        help="The discrete sampling frequency (F_s) in Hz. Default is 32000.0."
    )
    parser.add_argument(
        '--nfft', 
        type=int, 
        default=1024,
        help="The Fast Fourier Transform (FFT) segment length. Default is 1024."
    )
    return parser.parse_args()


def compute_spectral_statistics(
    time_series: npt.NDArray[np.complex64],
    sample_rate: float,
    n_per_segment: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""
    Compute the empirical expected Power Spectral Density (PSD) and its 
    variance over multiple independent temporal segments.
    """
    if sample_rate <= 0.0:
        raise ValueError("Sample rate must be strictly positive.")
    if time_series.size < n_per_segment:
        raise ValueError(f"Time series length ({time_series.size}) is insufficient for the specified FFT window.")

    # CRITICAL FIX: detrend=False preserves the native DC component (0 Hz) of the 
    # complex baseband spectrum, preventing artificial singularity notches.
    
    # CRITICAL FIX: noverlap=0 ensures that all STFT segments are strictly 
    # independent in the time domain. This guarantees the mathematical validity 
    # of the 1/K variance reduction scaling for the Mean PSD estimator.
    frequencies, _, periodogram_matrix = signal.spectrogram(
        x=time_series,
        fs=sample_rate,
        window='blackmanharris',
        nperseg=n_per_segment,
        noverlap=0,                  # <-- Set this explicitly to 0
        return_onesided=False,
        scaling='density',
        mode='psd',
        detrend=False
    )

    frequencies = np.fft.fftshift(frequencies)
    periodogram_matrix = np.fft.fftshift(periodogram_matrix, axes=0)

    # Number of temporal segments (K)
    k_segments = periodogram_matrix.shape[1]

    # Compute expected mean PSD
    expected_psd = np.mean(periodogram_matrix, axis=1)
    
    # Compute standard variance of the population
    sample_variance = np.var(periodogram_matrix, axis=1, ddof=1)
    
    # Scale by 1/K to obtain the variance of the Mean PSD estimator
    variance_mean_psd = sample_variance / k_segments

    return frequencies, expected_psd, variance_mean_psd


def main() -> None:
    r"""
    Primary execution sequence for empirical SDR data ingestion and preprocessing.
    """
    args = parse_arguments()

    if not args.input.is_file():
        sys.stderr.write(f"Error: Raw binary data file {args.input} cannot be located.\n")
        sys.exit(1)

    try:
        sys.stdout.write(f"Ingesting raw complex binary stream from {args.input}...\n")
        
        # Utilize memory-mapping to prevent RAM overflow when processing massive 
        # binary payloads (e.g., > 100 MB).
        raw_iq_data = np.memmap(args.input, dtype=np.complex64, mode='r')
        
        if raw_iq_data.size == 0:
            raise ValueError("The provided binary file contains no parseable complex64 data.")

        sys.stdout.write(f"Successfully mapped {raw_iq_data.size} complex samples.\n")
        sys.stdout.write("Executing STFT periodogram vectorization...\n")

        frequencies, mean_psd, var_psd = compute_spectral_statistics(
            time_series=raw_iq_data,
            sample_rate=args.fs,
            n_per_segment=args.nfft
        )

        dataset = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Power_Spectral_Density': mean_psd,
            'Variance': var_psd
        })

        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.output, index=False)

        sys.stdout.write("SDR preprocessing successfully executed.\n")
        sys.stdout.write(f"Output spectral resolution: {frequencies.size} bins.\n")
        sys.stdout.write(f"Layer 1 dataset serialized to: {args.output}\n")

    except Exception as e:
        sys.stderr.write(f"Execution aborted due to computational or IO error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()