import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHypothesis:
    """
    Immutable data structure encapsulating the definitions and constraints 
    of a specific statistical model within the trans-dimensional hypothesis space.
    """
    model_id: str
    log_likelihood: Callable[[npt.NDArray[np.float64]], float]
    prior_bounds: npt.NDArray[np.float64]
    log_scale_flags: npt.NDArray[np.bool_]

    @property
    def dimensionality(self) -> int:
        """Return the parameter space dimensionality of the hypothesis."""
        return self.prior_bounds.shape[0]


class GeneralizedReversibleJumpMCMC:
    r"""
    Execution engine for generalized Trans-dimensional Reversible Jump Markov 
    Chain Monte Carlo (RJ-MCMC) as formulated by Green (1995).

    This sampler constructs an ergodic Markov Chain capable of autonomous 
    transitions across an arbitrary number of competing models with disparate 
    dimensionalities, preserving detailed balance via auxiliary parameter sampling.

    Theoretical Formulation:
        The acceptance probability for a trans-dimensional jump from model M_i 
        to model M_j is given by:
        \alpha(i \to j) = \min\left(1, \frac{\mathcal{L}_j(\theta_j) \pi_j(\theta_j) q(i \mid j)}{\mathcal{L}_i(\theta_i) \pi_i(\theta_i) q(j \mid i) q(u)} \left| \frac{\partial \theta_j}{\partial (\theta_i, u)} \right| \right)

        Assuming symmetric inter-model proposal probabilities, nested parameter 
        spaces, and auxiliary parameters u drawn directly from the prior \pi(u), 
        the ratio simplifies to the likelihood ratio and evaluating the strict 
        prior boundaries for the shared parameter subset.
    """

    def __init__(
        self,
        hypotheses: List[ModelHypothesis],
        random_seed: int = 42
    ) -> None:
        """
        Initialize the generalized Reversible Jump MCMC engine.

        Parameters
        ----------
        hypotheses : List[ModelHypothesis]
            A sequence of initialized ModelHypothesis structures defining the 
            competing parameter spaces.
        random_seed : int, optional
            The stochastic initialization seed to guarantee deterministic 
            reproducibility. Default is 42.

        Raises
        ------
        ValueError
            If fewer than two competing hypotheses are provided.
        """
        if len(hypotheses) < 2:
            raise ValueError("RJ-MCMC requires a minimum of two competing hypotheses to establish trans-dimensional jumps.")

        self._hypotheses = hypotheses
        self._n_models = len(self._hypotheses)
        self._rng = np.random.default_rng(seed=random_seed)

        # Precompute standard deviations for intra-model random walk proposals (1% of prior domain)
        self._rw_sigma: Dict[int, npt.NDArray[np.float64]] = {}
        for idx, hyp in enumerate(self._hypotheses):
            self._rw_sigma[idx] = (hyp.prior_bounds[:, 1] - hyp.prior_bounds[:, 0]) * 0.01

    def _sample_auxiliary_parameters(
        self, 
        target_hypothesis: ModelHypothesis, 
        start_dim: int
    ) -> npt.NDArray[np.float64]:
        r"""
        Draw auxiliary parameters u exclusively for the extended dimensions 
        directly from their respective prior probability densities.

        Parameters
        ----------
        target_hypothesis : ModelHypothesis
            The destination model structure.
        start_dim : int
            The index at which the new dimensions begin.

        Returns
        -------
        npt.NDArray[np.float64]
            The independently sampled auxiliary parameter vector.
        """
        n_new_dims = target_hypothesis.dimensionality - start_dim
        u = np.empty(n_new_dims, dtype=np.float64)
        
        bounds_ext = target_hypothesis.prior_bounds[start_dim:]
        flags_ext = target_hypothesis.log_scale_flags[start_dim:]

        for i in range(n_new_dims):
            if flags_ext[i]:
                log_lower = np.log(bounds_ext[i, 0])
                log_upper = np.log(bounds_ext[i, 1])
                u[i] = np.exp(self._rng.uniform(log_lower, log_upper))
            else:
                u[i] = self._rng.uniform(bounds_ext[i, 0], bounds_ext[i, 1])
        return u

    def _evaluate_prior_support(
        self, 
        theta: npt.NDArray[np.float64], 
        bounds: npt.NDArray[np.float64]
    ) -> bool:
        """
        Evaluate physical constraints to instantaneously reject proposals 
        falling outside the defined prior domain.
        """
        return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))

    def _propose_intra_model_move(
        self, 
        model_idx: int, 
        current_theta: npt.NDArray[np.float64], 
        current_log_l: float
    ) -> Tuple[npt.NDArray[np.float64], float, int]:
        r"""
        Execute a standard Metropolis-Hastings Random Walk proposal within 
        the established parameter space dimensions.
        """
        hypothesis = self._hypotheses[model_idx]
        theta_prop = self._rng.normal(current_theta, self._rw_sigma[model_idx])
        
        if not self._evaluate_prior_support(theta_prop, hypothesis.prior_bounds):
            return current_theta, current_log_l, 0
            
        log_l_prop = hypothesis.log_likelihood(theta_prop)

        # Standard Metropolis acceptance criteria
        log_alpha = log_l_prop - current_log_l
        if np.log(self._rng.uniform()) < log_alpha:
            return theta_prop, log_l_prop, 1
        return current_theta, current_log_l, 0

    def _propose_inter_model_jump(
        self, 
        current_model_idx: int, 
        current_theta: npt.NDArray[np.float64], 
        current_log_l: float
    ) -> Tuple[int, npt.NDArray[np.float64], float, int]:
        r"""
        Propose a trans-dimensional state transition to an alternative hypothesis.

        The target model is selected uniformly from the remaining K-1 models. 
        Parameters are either truncated (death), mapped identically (isomorphic), 
        or augmented via prior sampling (birth).
        """
        available_indices = [i for i in range(self._n_models) if i != current_model_idx]
        target_model_idx = int(self._rng.choice(available_indices))
        
        target_hyp = self._hypotheses[target_model_idx]
        current_hyp = self._hypotheses[current_model_idx]

        dim_curr = current_hyp.dimensionality
        dim_targ = target_hyp.dimensionality
        dim_shared = min(dim_curr, dim_targ)

        # Truncate or map the shared dimensional subspace
        theta_shared = current_theta[:dim_shared]

        # Rigorous check: Ensure the shared parameters reside within the target's prior domain
        if not self._evaluate_prior_support(theta_shared, target_hyp.prior_bounds[:dim_shared]):
            return current_model_idx, current_theta, current_log_l, 0

        # Construct the proposed parameter vector
        if dim_targ > dim_curr:
            # Birth sequence: Sample auxiliary parameters
            u_ext = self._sample_auxiliary_parameters(target_hyp, dim_shared)
            theta_prop = np.concatenate([theta_shared, u_ext])
        else:
            # Death sequence or Isomorphic map
            theta_prop = theta_shared

        log_l_prop = target_hyp.log_likelihood(theta_prop)
        
        # Acceptance logic: The proposal and auxiliary prior densities mutually cancel
        log_alpha = log_l_prop - current_log_l

        if np.log(self._rng.uniform()) < log_alpha:
            return target_model_idx, theta_prop, log_l_prop, 1
        
        return current_model_idx, current_theta, current_log_l, 0

    def execute(self, n_iterations: int = 100000, burn_in: int = 20000) -> Dict[str, Any]:
        """
        Execute the generalized trans-dimensional Markov Chain Monte Carlo sequence.

        Parameters
        ----------
        n_iterations : int
            The total number of chain evaluations.
        burn_in : int
            The initial sequence of states to discard prior to reaching stationarity.

        Returns
        -------
        Dict[str, Any]
            The serialized traces of the model indicators, accepted parameter states 
            partitioned by model ID, and acceptance metrics.
        """
        model_indicator = np.zeros(n_iterations, dtype=np.int32)
        
        # Initialize chain at the center of the lowest-dimensional hypothesis
        current_model_idx = 0
        current_theta = np.mean(self._hypotheses[current_model_idx].prior_bounds, axis=1)
        current_log_l = self._hypotheses[current_model_idx].log_likelihood(current_theta)

        jump_attempts = 0
        jump_accepts = 0

        traces: Dict[str, List[npt.NDArray[np.float64]]] = {
            hyp.model_id: [] for hyp in self._hypotheses
        }

        for i in range(n_iterations):
            # Uniformly distribute proposal probability (50% Intra, 50% Inter)
            if self._rng.uniform() < 0.5:
                current_theta, current_log_l, _ = self._propose_intra_model_move(
                    current_model_idx, current_theta, current_log_l
                )
            else:
                jump_attempts += 1
                current_model_idx, current_theta, current_log_l, acc = self._propose_inter_model_jump(
                    current_model_idx, current_theta, current_log_l
                )
                jump_accepts += acc

            model_indicator[i] = current_model_idx
            
            if i >= burn_in:
                active_model_id = self._hypotheses[current_model_idx].model_id
                traces[active_model_id].append(current_theta.copy())

        # Finalize output trace structures
        output_traces = {k: np.array(v) for k, v in traces.items()}

        return {
            "model_indicator": model_indicator[burn_in:],
            "traces": output_traces,
            "jump_acceptance_rate": jump_accepts / max(1, jump_attempts)
        }