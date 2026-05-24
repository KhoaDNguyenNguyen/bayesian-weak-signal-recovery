import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


class ResidualDiagnostics:
    """
    Computes statistical diagnostics to evaluate the goodness-of-fit and 
    falsify the baseline noise assumptions utilizing raw residuals and 
    Locally Weighted Scatterplot Smoothing (LOWESS).

    Theoretical Formulation:
        Raw Residuals:
            r_i = D_i - P(f_i, \\theta_{MAP})
        Reduced Chi-Squared (\\chi^2 / \\nu):
            \\chi^2_\\nu = \\frac{1}{N - M} \\sum_{i=1}^{N} \\frac{r_i^2}{\\sigma_i^2}
        where N is the number of independent observations and M is the 
        dimensionality of the parameter space.
    """

    def __init__(
        self,
        observational_data: npt.NDArray[np.float64],
        model_prediction: npt.NDArray[np.float64],
        noise_variance: npt.NDArray[np.float64],
        n_parameters: int
    ) -> None:
        """
        Initialize the diagnostic evaluation engine.

        Parameters
        ----------
        observational_data : npt.NDArray[np.float64]
            The 1-dimensional array of empirical data points (D_i).
        model_prediction : npt.NDArray[np.float64]
            The 1-dimensional array of deterministic model predictions evaluated 
            at the Maximum A Posteriori (MAP) estimate (P_i).
        noise_variance : npt.NDArray[np.float64]
            The 1-dimensional array of strictly positive observational variances (\\sigma_i^2).
        n_parameters : int
            The total number of free parameters in the physical model (M).

        Raises
        ------
        ValueError
            If input arrays exhibit dimensional mismatches, if variance contains 
            non-positive values, or if the degrees of freedom are non-positive.
        """
        if not (observational_data.shape == model_prediction.shape == noise_variance.shape):
            raise ValueError("Dimensionality mismatch across observational, prediction, and variance arrays.")
        if np.any(noise_variance <= 0.0):
            raise ValueError("Noise variance must be strictly positive definite.")
        if n_parameters < 0:
            raise ValueError("The number of model parameters must be non-negative.")
        
        self._n_observations = observational_data.size
        self._degrees_of_freedom = self._n_observations - n_parameters

        if self._degrees_of_freedom <= 0:
            raise ValueError(
                f"Non-positive degrees of freedom (N={self._n_observations}, "
                f"M={n_parameters}). The system is underdetermined."
            )

        self._data = observational_data
        self._model = model_prediction
        self._variance = noise_variance
        
        # Isolate the raw structural deviations
        self._residuals = self._data - self._model

    @property
    def raw_residuals(self) -> npt.NDArray[np.float64]:
        """
        Retrieve the computed raw residuals.

        Returns
        -------
        npt.NDArray[np.float64]
            The 1-dimensional residual array.
        """
        return self._residuals

    def compute_reduced_chi_squared(self) -> float:
        """
        Compute the reduced Chi-squared statistic for the MAP model.

        Returns
        -------
        float
            The scalar value of \\chi^2 / \\nu. Values significantly deviating 
            from 1.0 indicate systematic discrepancies or mischaracterized variance.
        """
        chi_squared = np.sum(np.square(self._residuals) / self._variance)
        return float(chi_squared / self._degrees_of_freedom)

    def compute_lowess_trend(
        self, 
        independent_variable: npt.NDArray[np.float64], 
        fraction: float = 0.2
    ) -> npt.NDArray[np.float64]:
        """
        Execute Locally Weighted Scatterplot Smoothing (LOWESS) to extract 
        non-linear structural trends from the raw residuals.

        This non-parametric regression technique identifies systematic background 
        inadequacies (e.g., residual continuum emission, correlated red noise).

        Parameters
        ----------
        independent_variable : npt.NDArray[np.float64]
            The 1-dimensional array of independent variables (e.g., frequencies).
        fraction : float, optional
            The bandwidth parameter defining the fraction of the data utilized 
            to compute each local regression fit. Default is 0.2.

        Returns
        -------
        npt.NDArray[np.float64]
            The 1-dimensional array representing the isolated continuous trend.

        Raises
        ------
        ValueError
            If the smoothing fraction is outside the open interval (0, 1].
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("The LOWESS fractional bandwidth must reside in the interval (0, 1].")

        # Utilize statsmodels implementation for mathematically rigorous tricube weighting
        smoothed_trend = sm.nonparametric.lowess(
            self._residuals,
            independent_variable,
            frac=fraction,
            return_sorted=False
        )
        
        return smoothed_trend