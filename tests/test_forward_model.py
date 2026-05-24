import pytest
import numpy as np
import numpy.typing as npt

from bwsr_inference.forward_model.signal import GaussianSignal, VoigtSignal
from bwsr_inference.forward_model.background import ExponentialBackground, PolynomialBackground


@pytest.fixture
def frequency_domain() -> npt.NDArray[np.float64]:
    """
    Generate a standard high-resolution independent variable array for testing.
    """
    return np.linspace(0.0, 100.0, 1000, dtype=np.float64)


def test_gaussian_signal_mathematical_correctness(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the numerical output of the Gaussian signal model against the 
    analytical limit, specifically checking the peak amplitude evaluation.
    """
    model = GaussianSignal()
    amplitude = 5.0
    center_frequency = 50.0
    standard_deviation = 2.0
    theta = np.array([amplitude, center_frequency, standard_deviation], dtype=np.float64)

    evaluation = model(frequency_domain, theta)

    # Find the index closest to the center frequency
    center_index = np.argmin(np.abs(frequency_domain - center_frequency))
    
    # The evaluation at the exact center frequency must approach the amplitude parameter
    np.testing.assert_allclose(
        evaluation[center_index], 
        amplitude, 
        rtol=1e-3, 
        err_msg="Gaussian peak evaluation deviates from the analytical limit."
    )


def test_gaussian_signal_physical_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Ensure the Gaussian model rigorously rejects unphysical structural parameters.
    """
    model = GaussianSignal()
    
    # Zero standard deviation must trigger a ValueError due to division by zero implications
    theta_zero_width = np.array([1.0, 50.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_zero_width)

    # Negative standard deviation represents an unphysical state
    theta_negative_width = np.array([1.0, 50.0, -1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_negative_width)


def test_gaussian_signal_dimensionality_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the parameter array dimensionality validation.
    """
    model = GaussianSignal()
    theta_insufficient = np.array([1.0, 50.0], dtype=np.float64)
    
    with pytest.raises(ValueError):
        model(frequency_domain, theta_insufficient)


def test_voigt_signal_physical_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Validate the broadening parameter boundaries for the pseudo-Voigt profile.
    """
    model = VoigtSignal()
    
    # Both Gaussian and Lorentzian broadening parameters cannot be simultaneously zero
    theta_zero_widths = np.array([1.0, 50.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_zero_widths)


def test_exponential_background_asymptotics(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the asymptotic behavior of the exponential continuum model.
    """
    model = ExponentialBackground()
    amplitude = 10.0
    decay_constant = 1.0  # Rapid decay
    offset = 2.5
    theta = np.array([amplitude, decay_constant, offset], dtype=np.float64)

    evaluation = model(frequency_domain, theta)

    # At f = 0, the model should precisely equal Amplitude + Offset
    np.testing.assert_allclose(evaluation[0], amplitude + offset, rtol=1e-7)

    # As f -> infinity, the model must converge to the baseline offset
    np.testing.assert_allclose(evaluation[-1], offset, rtol=1e-7)


def test_polynomial_background_correctness(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Validate the numerical stability and accuracy of the polynomial background 
    utilizing Horner's method implementation.
    """
    model = PolynomialBackground()
    # Coefficients: [c_0, c_1, c_2] -> c_0 + c_1*x + c_2*x^2
    theta = np.array([5.0, -0.5, 0.01], dtype=np.float64)

    evaluation = model(frequency_domain, theta)
    
    # Compute baseline comparison array via analytical standard formulation
    analytical_baseline = theta[0] + theta[1] * frequency_domain + theta[2] * (frequency_domain ** 2)

    np.testing.assert_allclose(
        evaluation, 
        analytical_baseline, 
        rtol=1e-7,
        err_msg="Polynomial evaluation utilizing Horner's method exhibits computational drift."
    )


def test_polynomial_background_empty_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Ensure the polynomial model restricts execution upon receiving an empty parameter vector.
    """
    model = PolynomialBackground()
    theta_empty = np.array([], dtype=np.float64)
    
    with pytest.raises(ValueError):
        model(frequency_domain, theta_empty)