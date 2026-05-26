import pytest
import numpy as np
import numpy.typing as npt

from bwsr_inference.forward_model import MODEL_ZOO


@pytest.fixture
def frequency_domain() -> npt.NDArray[np.float64]:
    """
    Generate a standard high-resolution independent variable array for testing.
    """
    return np.linspace(0.0, 100.0, 1000, dtype=np.float64)


def test_h0_polynomial_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the architectural constraints of the H0 null hypothesis model.
    """
    model = MODEL_ZOO['h0']
    
    # Empty parameter array must trigger a structural violation
    theta_empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_empty)

    # Valid evaluation with a single background constant
    theta_valid = np.array([5.0], dtype=np.float64)
    evaluation = model(frequency_domain, theta_valid)
    assert evaluation.shape == frequency_domain.shape


def test_h1_poly_gaussian_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the architectural parameter routing of the H1 Gaussian hypothesis.
    """
    model = MODEL_ZOO['h1_gauss']
    
    # Insufficient parameters (requires at least 1 background + 3 signal = 4)
    theta_insufficient = np.array([5.0, 1.0, 50.0], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_insufficient)

    # Valid evaluation (Constant background + Gaussian signal)
    theta_valid = np.array([5.0, 2.0, 50.0, 1.5], dtype=np.float64)
    evaluation = model(frequency_domain, theta_valid)
    assert evaluation.shape == frequency_domain.shape


def test_h1_poly_voigt_constraints(frequency_domain: npt.NDArray[np.float64]) -> None:
    """
    Verify the architectural parameter routing of the H1 Voigt hypothesis.
    """
    model = MODEL_ZOO['h1_voigt']
    
    # Insufficient parameters (requires at least 1 background + 4 signal = 5)
    theta_insufficient = np.array([5.0, 2.0, 50.0, 1.5], dtype=np.float64)
    with pytest.raises(ValueError):
        model(frequency_domain, theta_insufficient)

    # Valid evaluation (Constant background + Voigt signal)
    theta_valid = np.array([5.0, 2.0, 50.0, 1.5, 0.5], dtype=np.float64)
    evaluation = model(frequency_domain, theta_valid)
    assert evaluation.shape == frequency_domain.shape