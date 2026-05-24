import numpy as np
from typing import Dict, Any, Union
import scipy.special
from dataclasses import dataclass


@dataclass(frozen=True)
class SignificanceMetrics:
    """
    Immutable data structure encapsulating the computed statistical 
    significance metrics derived from the Look-Elsewhere Effect evaluation.
    """
    local_significance: float
    local_p_value: float
    trials_factor: float
    global_p_value: float
    global_significance: float
    is_confirmed_detection: bool


class GlobalSignificanceEvaluator:
    """
    Computes the global statistical significance of a local detection by 
    correcting for the multiple testing problem (Look-Elsewhere Effect) 
    over a continuous parameter space.

    The evaluation utilizes the Gross-Vitells (2010) formulation for the 
    expected number of upcrossings of a stationary Gaussian random field.

    Theoretical Formulation:
        Trials Factor:
            N_{trials} = \frac{\Delta f}{\delta f}
        Local p-value:
            p_{local} = \frac{1}{2} \text{erfc}\left( \frac{Z_{local}}{\sqrt{2}} \right)
        Global p-value (Gross-Vitells Bound):
            p_{global} \approx \min\left(1.0, p_{local} + N_{trials} \cdot \exp\left(-\frac{Z_{local}^2}{2}\right)\right)
        Global Significance:
            Z_{global} = \sqrt{2} \cdot \text{erfcinv}(2 \cdot p_{global})
    """

    # Strict confidence threshold for definitive physical detection (5-sigma standard)
    DETECTION_THRESHOLD_SIGMA: float = 5.0

    def __init__(
        self,
        search_bandwidth: float,
        signal_resolution: float
    ) -> None:
        """
        Initialize the evaluator with domain constraints to precompute the trials factor.

        Parameters
        ----------
        search_bandwidth : float
            The total frequency range scanned during the inference procedure (\Delta f).
        signal_resolution : float
            The characteristic width or resolution element of the target signal (\delta f).

        Raises
        ------
        ValueError
            If bandwidth or resolution are non-positive, or if the resolution 
            exceeds the entire search bandwidth.
        """
        if search_bandwidth <= 0.0:
            raise ValueError("Search bandwidth must be strictly positive.")
        if signal_resolution <= 0.0:
            raise ValueError("Signal resolution must be strictly positive.")
        if signal_resolution > search_bandwidth:
            raise ValueError("Signal resolution cannot exceed the total search bandwidth.")

        self._search_bandwidth = search_bandwidth
        self._signal_resolution = signal_resolution
        
        # Determine the number of independent resolution elements
        self._trials_factor = self._search_bandwidth / self._signal_resolution

    def _compute_local_p_value(self, z_local: float) -> float:
        """
        Compute the local p-value corresponding to the local significance.

        Parameters
        ----------
        z_local : float
            The local significance level (in standard deviations, \sigma).

        Returns
        -------
        float
            The local probability of observing a background fluctuation 
            at least as extreme as the detection.
        """
        return 0.5 * scipy.special.erfc(z_local / np.sqrt(2.0))

    def _compute_global_p_value(self, z_local: float, p_local: float) -> float:
        """
        Evaluate the Gross-Vitells global p-value bound.

        Parameters
        ----------
        z_local : float
            The local significance level.
        p_local : float
            The local p-value.

        Returns
        -------
        float
            The bounded global p-value, capped strictly at 1.0.
        """
        # Formulate the asymptotic upcrossing expectation
        upcrossing_expectation = self._trials_factor * np.exp(-0.5 * (z_local ** 2))
        
        p_global = p_local + upcrossing_expectation
        return min(1.0, float(p_global))

    def _compute_global_significance(self, p_global: float) -> float:
        """
        Convert the global p-value back into a Gaussian significance equivalent.

        Parameters
        ----------
        p_global : float
            The computed global p-value.

        Returns
        -------
        float
            The global significance level in standard deviations (\sigma). 
            Returns 0.0 if the corrected p-value is > 0.5. Returns np.inf 
            if the p-value strictly underflows representable floating point precision.
        """
        if p_global >= 0.5:
            return 0.0
        
        if p_global == 0.0:
            return float(np.inf)

        return np.sqrt(2.0) * scipy.special.erfcinv(2.0 * p_global)

    def evaluate(self, local_significance: float) -> SignificanceMetrics:
        """
        Execute the Look-Elsewhere Effect correction sequence.

        Parameters
        ----------
        local_significance : float
            The local likelihood ratio expressed as an equivalent Gaussian 
            standard deviation (Z_{local}).

        Returns
        -------
        SignificanceMetrics
            A strongly typed, immutable dataclass containing the complete 
            statistical evaluation hierarchy.

        Raises
        ------
        ValueError
            If the local significance is negative.
        """
        if local_significance < 0.0:
            raise ValueError("Local significance must be a non-negative scalar.")

        p_local = self._compute_local_p_value(local_significance)
        p_global = self._compute_global_p_value(local_significance, p_local)
        z_global = self._compute_global_significance(p_global)

        is_significant = bool(z_global >= self.DETECTION_THRESHOLD_SIGMA)

        return SignificanceMetrics(
            local_significance=float(local_significance),
            local_p_value=float(p_local),
            trials_factor=float(self._trials_factor),
            global_p_value=float(p_global),
            global_significance=float(z_global),
            is_confirmed_detection=is_significant
        )