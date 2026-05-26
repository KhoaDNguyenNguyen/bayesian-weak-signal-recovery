import pytest
import numpy as np
import numpy.typing as npt

from bwsr_inference.sampling.rjmcmc import ReversibleJumpMCMC


def mock_likelihood_h0(theta: npt.NDArray[np.float64]) -> float:
    """Mock baseline likelihood mapping."""
    return -10.0


def mock_likelihood_h1(theta: npt.NDArray[np.float64]) -> float:
    """Mock alternative likelihood mapping favoring the signal space."""
    return -5.0


@pytest.fixture
def rjmcmc_context() -> ReversibleJumpMCMC:
    """
    Construct a deterministic Reversible Jump MCMC engine context.
    H0 operates in R^1, H1 operates in R^3.
    """
    bounds_h0 = np.array([[0.0, 1.0]], dtype=np.float64)
    bounds_h1 = np.array([
        [0.0, 1.0],     # Background
        [-5.0, 5.0],    # Signal Parameter 1
        [1.0, 10.0]     # Signal Parameter 2
    ], dtype=np.float64)
    
    flags_h1 = np.array([False, False, True], dtype=np.bool_)

    return ReversibleJumpMCMC(
        likelihood_h0=mock_likelihood_h0,
        likelihood_h1=mock_likelihood_h1,
        prior_bounds_h0=bounds_h0,
        prior_bounds_h1=bounds_h1,
        log_scale_flags_h1=flags_h1,
        random_seed=12345
    )


def test_rjmcmc_initialization_constraints() -> None:
    """
    Verify the trans-dimensional structural constraint enforcement.
    """
    bounds_h0 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    bounds_h1 = np.array([[0.0, 1.0]], dtype=np.float64) # H1 dimensionality < H0 dimensionality
    flags_h1 = np.array([False], dtype=np.bool_)

    with pytest.raises(ValueError):
        ReversibleJumpMCMC(
            likelihood_h0=mock_likelihood_h0,
            likelihood_h1=mock_likelihood_h1,
            prior_bounds_h0=bounds_h0,
            prior_bounds_h1=bounds_h1,
            log_scale_flags_h1=flags_h1
        )


def test_trans_dimensional_birth_move(rjmcmc_context: ReversibleJumpMCMC) -> None:
    """
    Verify the geometry expansion during a Birth move (H0 -> H1).
    """
    theta_h0 = np.array([0.5], dtype=np.float64)
    log_l_h0 = mock_likelihood_h0(theta_h0)

    # Force RNG to yield a specific value for deterministic testing
    model_id, theta_prop, log_l_prop, _ = rjmcmc_context._trans_dimensional_birth(theta_h0, log_l_h0)

    # The proposed theta must possess the exact dimensionality of H1 (R^3)
    assert theta_prop.size == rjmcmc_context._n_dim_h1
    
    # The baseline background parameter must remain immutable during the jump
    np.testing.assert_allclose(theta_prop[0], theta_h0[0], rtol=1e-7)


def test_trans_dimensional_death_move(rjmcmc_context: ReversibleJumpMCMC) -> None:
    """
    Verify the geometry truncation during a Death move (H1 -> H0).
    """
    theta_h1 = np.array([0.5, 2.0, 5.0], dtype=np.float64)
    
    # Overwrite the likelihood dynamically to GUARANTEE acceptance of the death move.
    # If ln(L_H0) is exceptionally high (0.0) compared to ln(L_H1) (-10.0), 
    # the Metropolis-Hastings acceptance ratio evaluates to e^(10), 
    # ensuring absolute acceptance without relying on stochasticity.
    rjmcmc_context._log_l_h0 = lambda t: 0.0
    rjmcmc_context._log_l_h1 = lambda t: -10.0
    
    log_l_h1 = rjmcmc_context._log_l_h1(theta_h1)

    model_id, theta_prop, log_l_prop, accepted = rjmcmc_context._trans_dimensional_death(theta_h1, log_l_h1)

    # Strictly assert that the Markov chain accepted the state transition
    assert accepted == 1, "The Metropolis-Hastings criteria erroneously rejected an optimal death move."

    # The proposed theta must truncate down to the exact dimensionality of H0 (R^1)
    assert theta_prop.size == rjmcmc_context._n_dim_h0
    
    # The surviving baseline parameter must strictly match the pre-truncation state
    np.testing.assert_allclose(theta_prop[0], theta_h1[0], rtol=1e-7)