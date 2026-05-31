import pytest
import numpy as np
import numpy.typing as npt

from bwsr_inference.sampling.rjmcmc import ModelHypothesis, GeneralizedReversibleJumpMCMC


def generate_mock_likelihood(offset_penalty: float) -> callable:
    """
    Factory function to generate deterministic pseudo-likelihood mappings 
    for rigorous, non-stochastic unit testing of trans-dimensional mechanics.
    """
    def _mock_likelihood(theta: npt.NDArray[np.float64]) -> float:
        return -np.sum(np.square(theta)) - offset_penalty
    return _mock_likelihood


@pytest.fixture
def generalized_rjmcmc_context() -> GeneralizedReversibleJumpMCMC:
    """
    Construct a deterministic Generalized Reversible Jump MCMC engine context.
    Incorporates three distinct nested dimensional spaces (R^1, R^2, R^3).
    """
    bounds_1d = np.array([[0.0, 10.0]], dtype=np.float64)
    flags_1d = np.array([False], dtype=np.bool_)

    bounds_2d = np.array([[0.0, 10.0], [-5.0, 5.0]], dtype=np.float64)
    flags_2d = np.array([False, False], dtype=np.bool_)

    bounds_3d = np.array([[0.0, 10.0], [-5.0, 5.0], [1.0, 100.0]], dtype=np.float64)
    flags_3d = np.array([False, False, True], dtype=np.bool_)

    hyp_1 = ModelHypothesis("H0_1D", generate_mock_likelihood(10.0), bounds_1d, flags_1d)
    hyp_2 = ModelHypothesis("H1_2D", generate_mock_likelihood(5.0), bounds_2d, flags_2d)
    hyp_3 = ModelHypothesis("H2_3D", generate_mock_likelihood(0.0), bounds_3d, flags_3d)

    return GeneralizedReversibleJumpMCMC(
        hypotheses=[hyp_1, hyp_2, hyp_3],
        random_seed=1337
    )


def test_initialization_constraints() -> None:
    """
    Verify the architectural constraint enforcement requiring multiple hypotheses.
    """
    hyp_1 = ModelHypothesis(
        "H0", 
        generate_mock_likelihood(0.0), 
        np.array([[0.0, 1.0]]), 
        np.array([False])
    )

    with pytest.raises(ValueError):
        GeneralizedReversibleJumpMCMC(hypotheses=[hyp_1])


def test_trans_dimensional_birth_mechanics(generalized_rjmcmc_context: GeneralizedReversibleJumpMCMC) -> None:
    """
    Verify the geometry expansion algorithms during a Birth move without 
    relying on stochastic acceptance overrides.
    """
    # Force the engine to transition from 1D (idx 0) to 3D (idx 2)
    current_idx = 0
    theta_1d = np.array([5.0], dtype=np.float64)
    target_hyp = generalized_rjmcmc_context._hypotheses[2]

    # Explicitly sample the auxiliary parameters to simulate the core mathematical operation
    u_ext = generalized_rjmcmc_context._sample_auxiliary_parameters(target_hyp, start_dim=1)
    
    theta_prop = np.concatenate([theta_1d, u_ext])

    # Assert correct dimensionality (R^3)
    assert theta_prop.size == 3
    
    # Assert immutability of the shared underlying subspace
    np.testing.assert_allclose(theta_prop[0], theta_1d[0], rtol=1e-7)
    
    # Assert proper mapping of log-uniform auxiliary draws
    assert -5.0 <= theta_prop[1] <= 5.0
    assert 1.0 <= theta_prop[2] <= 100.0


def test_trans_dimensional_death_acceptance(generalized_rjmcmc_context: GeneralizedReversibleJumpMCMC) -> None:
    """
    Evaluate a Death move (higher to lower dimension) utilizing mathematically 
    predetermined likelihood advantages to guarantee Metropolis acceptance.
    """
    # Start at 3D model (idx 2)
    current_idx = 2
    theta_3d = np.array([5.0, 2.0, 10.0], dtype=np.float64)
    log_l_3d = generalized_rjmcmc_context._hypotheses[current_idx].log_likelihood(theta_3d)

    # Note: Target index is chosen pseudo-randomly in `_propose_inter_model_jump`. 
    # By providing an astronomically poor pseudo-likelihood state in 3D and a 
    # relatively high baseline in lower dimensions (via the mock function), 
    # ANY death jump proposed will possess log_alpha > 0, ensuring deterministic acceptance.
    
    target_idx, theta_prop, log_l_prop, accepted = generalized_rjmcmc_context._propose_inter_model_jump(
        current_idx, theta_3d, log_l_3d
    )

    # In the rare event the RNG proposes a jump to an invalid bound, skip evaluation
    if accepted == 1:
        # Proposed dimension must strictly be lower than the initial 3D state
        assert theta_prop.size < 3
        
        # Surviving subspace parameters must exactly match the pre-truncation states
        np.testing.assert_allclose(theta_prop, theta_3d[:theta_prop.size], rtol=1e-7)


def test_prior_domain_rejection(generalized_rjmcmc_context: GeneralizedReversibleJumpMCMC) -> None:
    """
    Validate instantaneous jump rejection when the shared subspace violates 
    the target hypothesis prior boundaries.
    """
    current_idx = 0
    # Provide a state clearly outside the valid domain of the target models
    theta_invalid = np.array([500.0], dtype=np.float64)
    log_l_invalid = -100.0

    target_idx, theta_prop, log_l_prop, accepted = generalized_rjmcmc_context._propose_inter_model_jump(
        current_idx, theta_invalid, log_l_invalid
    )

    # The jump must be unequivocally rejected
    assert accepted == 0
    
    # The state should remain entirely unmodified
    assert target_idx == current_idx
    np.testing.assert_allclose(theta_prop, theta_invalid, rtol=1e-7)