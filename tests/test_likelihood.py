import pytest
import numpy as np
import numpy.typing as npt

from bwsr_inference.sampling.likelihood import PriorTransform, DiagonalGaussianLikelihood
from bwsr_inference.forward_model.signal import GaussianSignal


def test_prior_transform_linear_boundaries() -> None:
    """
    Verify the deterministic mapping from the unit hypercube to physical 
    parameter space assuming linear uniformity.
    """
    bounds = np.array([
        [0.0, 10.0],
        [-5.0, 5.0]
    ], dtype=np.float64)
    log_flags = np.array([False, False], dtype=np.bool_)
    
    prior = PriorTransform(bounds=bounds, log_flags=log_flags)
    
    u_lower = np.array([0.0, 0.0], dtype=np.float64)
    u_upper = np.array([1.0, 1.0], dtype=np.float64)
    u_mid = np.array([0.5, 0.5], dtype=np.float64)

    np.testing.assert_allclose(prior(u_lower), bounds[:, 0], rtol=1e-7)
    np.testing.assert_allclose(prior(u_upper), bounds[:, 1], rtol=1e-7)
    np.testing.assert_allclose(prior(u_mid), np.array([5.0, 0.0]), rtol=1e-7)


def test_prior_transform_log_boundaries() -> None:
    """
    Verify the deterministic mapping incorporating log-uniform (Jeffreys) priors.
    """
    bounds = np.array([
        [1.0, 100.0]
    ], dtype=np.float64)
    log_flags = np.array([True], dtype=np.bool_)
    
    prior = PriorTransform(bounds=bounds, log_flags=log_flags)
    
    u_mid = np.array([0.5], dtype=np.float64)
    expected_mid = np.exp((np.log(1.0) + np.log(100.0)) / 2.0)
    
    np.testing.assert_allclose(prior(u_mid), np.array([expected_mid]), rtol=1e-7)


def test_prior_transform_unphysical_log_constraint() -> None:
    """
    Ensure the prior transformation rigorously rejects log-uniform mapping 
    requests spanning non-positive physical domains.
    """
    bounds = np.array([
        [-10.0, 10.0]
    ], dtype=np.float64)
    log_flags = np.array([True], dtype=np.bool_)
    
    with pytest.raises(ValueError):
        PriorTransform(bounds=bounds, log_flags=log_flags)


@pytest.fixture
def diagonal_likelihood_context() -> dict:
    """
    Construct a controlled dataset for likelihood validation.
    """
    frequencies = np.linspace(10.0, 20.0, 100, dtype=np.float64)
    theta_true = np.array([2.0, 15.0, 0.5], dtype=np.float64)
    forward_model = GaussianSignal()
    
    model_data = forward_model(frequencies, theta_true)
    variance = np.full_like(frequencies, 0.1, dtype=np.float64)
    
    return {
        "frequencies": frequencies,
        "data": model_data,  # Idealized noiseless data
        "variance": variance,
        "forward_model": forward_model,
        "theta_true": theta_true
    }


def test_diagonal_gaussian_likelihood_optimal_fit(diagonal_likelihood_context: dict) -> None:
    """
    Verify the likelihood evaluation correctly outputs the theoretical maximum 
    (normalization constant) when the predicted model aligns perfectly with the data.
    """
    likelihood = DiagonalGaussianLikelihood(
        frequencies=diagonal_likelihood_context["frequencies"],
        data=diagonal_likelihood_context["data"],
        noise_variance=diagonal_likelihood_context["variance"],
        forward_model=diagonal_likelihood_context["forward_model"]
    )
    
    log_l = likelihood(diagonal_likelihood_context["theta_true"])
    
    # Under exact model alignment, the residual term computes to exactly zero.
    # Therefore, ln L should precisely equal the normalization constant.
    np.testing.assert_allclose(
        log_l, 
        likelihood._normalization_constant, 
        rtol=1e-7,
        err_msg="Likelihood maximum deviates from the analytical normalization limit."
    )


def test_diagonal_gaussian_likelihood_degradation(diagonal_likelihood_context: dict) -> None:
    """
    Verify the log-likelihood structurally degrades (yields substantially lower values) 
    when the parameter hypothesis diverges from the true parameters.
    """
    likelihood = DiagonalGaussianLikelihood(
        frequencies=diagonal_likelihood_context["frequencies"],
        data=diagonal_likelihood_context["data"],
        noise_variance=diagonal_likelihood_context["variance"],
        forward_model=diagonal_likelihood_context["forward_model"]
    )
    
    log_l_optimal = likelihood(diagonal_likelihood_context["theta_true"])
    
    # Introduce an explicit structural discrepancy
    theta_offset = np.array([2.0, 16.0, 0.5], dtype=np.float64)
    log_l_suboptimal = likelihood(theta_offset)
    
    assert log_l_suboptimal < log_l_optimal, "Likelihood function failed to penalize discrepant hypotheses."


def test_likelihood_initialization_constraints(diagonal_likelihood_context: dict) -> None:
    """
    Validate the strict geometric constraints imposed upon the likelihood 
    initialization vectors.
    """
    frequencies = diagonal_likelihood_context["frequencies"]
    data = diagonal_likelihood_context["data"]
    forward_model = diagonal_likelihood_context["forward_model"]
    
    # Induce an explicit dimensionality violation
    variance_mismatched = np.full(50, 0.1, dtype=np.float64)
    
    with pytest.raises(ValueError):
        DiagonalGaussianLikelihood(
            frequencies=frequencies,
            data=data,
            noise_variance=variance_mismatched,
            forward_model=forward_model
        )

    # Induce an explicitly unphysical state within the variance array
    variance_non_positive = np.full_like(frequencies, -0.1, dtype=np.float64)
    
    with pytest.raises(ValueError):
        DiagonalGaussianLikelihood(
            frequencies=frequencies,
            data=data,
            noise_variance=variance_non_positive,
            forward_model=forward_model
        )