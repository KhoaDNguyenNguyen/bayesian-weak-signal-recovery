import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Any, Tuple, List


class ReversibleJumpMCMC:
    r"""
    Execution engine for Trans-dimensional Reversible Jump Markov Chain Monte Carlo 
    (RJ-MCMC) as formulated by Green (1995).

    This sampler constructs an ergodic Markov Chain capable of transitioning 
    between competing models of disparate dimensionalities (e.g., jumping from 
    a background-only hypothesis H0 in R^N to a signal-present hypothesis H1 in R^{N+M}).
    """

    def __init__(
        self,
        likelihood_h0: Callable[[npt.NDArray[np.float64]], float],
        likelihood_h1: Callable[[npt.NDArray[np.float64]], float],
        prior_bounds_h0: npt.NDArray[np.float64],
        prior_bounds_h1: npt.NDArray[np.float64],
        log_scale_flags_h1: npt.NDArray[np.bool_],
        random_seed: int = 42
    ) -> None:
        """
        Initialize the Reversible Jump MCMC engine.

        Parameters
        ----------
        likelihood_h0 : Callable
            The log-likelihood function for the lower-dimensional null hypothesis.
        likelihood_h1 : Callable
            The log-likelihood function for the higher-dimensional alternative hypothesis.
        prior_bounds_h0 : npt.NDArray[np.float64]
            The boundaries for the H0 parameter space.
        prior_bounds_h1 : npt.NDArray[np.float64]
            The boundaries for the H1 parameter space.
        log_scale_flags_h1 : npt.NDArray[np.bool_]
            The boolean array indicating logarithmic scale mapping for H1 prior sampling.
        random_seed : int
            The stochastic initialization seed to guarantee reproducibility.
        """
        self._log_l_h0 = likelihood_h0
        self._log_l_h1 = likelihood_h1
        self._bounds_h0 = prior_bounds_h0
        self._bounds_h1 = prior_bounds_h1
        self._log_flags_h1 = log_scale_flags_h1

        self._n_dim_h0 = prior_bounds_h0.shape[0]
        self._n_dim_h1 = prior_bounds_h1.shape[0]
        self._n_dim_diff = self._n_dim_h1 - self._n_dim_h0

        if self._n_dim_diff <= 0:
            raise ValueError("H1 dimensionality must strictly exceed H0 dimensionality for trans-dimensional jumps.")

        self._rng = np.random.default_rng(seed=random_seed)

        # Inter-model proposal standard deviations (Random Walk step sizes)
        self._rw_sigma_h0 = (self._bounds_h0[:, 1] - self._bounds_h0[:, 0]) * 0.01
        self._rw_sigma_h1 = (self._bounds_h1[:, 1] - self._bounds_h1[:, 0]) * 0.01

    def _sample_prior_signal(self) -> npt.NDArray[np.float64]:
        r"""
        Draw auxiliary parameters $u$ exclusively for the transient signal components 
        directly from their respective prior probability densities.

        Returns
        -------
        npt.NDArray[np.float64]
            The independently sampled auxiliary parameter vector.
        """
        u = np.empty(self._n_dim_diff, dtype=np.float64)
        bounds_sig = self._bounds_h1[self._n_dim_h0:]
        flags_sig = self._log_flags_h1[self._n_dim_h0:]

        for i in range(self._n_dim_diff):
            if flags_sig[i]:
                log_lower = np.log(bounds_sig[i, 0])
                log_upper = np.log(bounds_sig[i, 1])
                u[i] = np.exp(self._rng.uniform(log_lower, log_upper))
            else:
                u[i] = self._rng.uniform(bounds_sig[i, 0], bounds_sig[i, 1])
        return u

    def _check_boundaries(self, theta: npt.NDArray[np.float64], bounds: npt.NDArray[np.float64]) -> bool:
        """
        Evaluate physical constraints to instantaneously reject out-of-bound proposals.
        """
        return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))

    def _intra_model_move(self, current_model: int, current_theta: npt.NDArray[np.float64], current_log_l: float) -> Tuple[npt.NDArray[np.float64], float, int]:
        r"""
        Execute a standard Metropolis-Hastings Random Walk proposal within 
        the established parameter space dimensions.
        """
        accepted = 0
        if current_model == 0:
            theta_prop = self._rng.normal(current_theta, self._rw_sigma_h0)
            if not self._check_boundaries(theta_prop, self._bounds_h0):
                return current_theta, current_log_l, accepted
            log_l_prop = self._log_l_h0(theta_prop)
        else:
            theta_prop = self._rng.normal(current_theta, self._rw_sigma_h1)
            if not self._check_boundaries(theta_prop, self._bounds_h1):
                return current_theta, current_log_l, accepted
            log_l_prop = self._log_l_h1(theta_prop)

        # Standard Metropolis acceptance criteria (Symmetric proposal implies q(theta|theta') / q(theta'|theta) = 1)
        log_alpha = log_l_prop - current_log_l
        if np.log(self._rng.uniform()) < log_alpha:
            return theta_prop, log_l_prop, 1
        return current_theta, current_log_l, accepted

    def _trans_dimensional_birth(self, current_theta: npt.NDArray[np.float64], current_log_l: float) -> Tuple[int, npt.NDArray[np.float64], float, int]:
        r"""
        Execute the 'Birth' move: Transitioning from H0 -> H1.

        Theoretical Formulation:
            \alpha(H_0 \to H_1) = \min(1, \frac{\mathcal{L}_1}{\mathcal{L}_0} \times |J|)
        Given that $u$ is drawn identically from the prior, the prior ratio and 
        proposal ratio mutually cancel. The identity mapping guarantees $|J| = 1$.
        """
        u_signal = self._sample_prior_signal()
        theta_prop = np.concatenate([current_theta, u_signal])
        
        log_l_prop = self._log_l_h1(theta_prop)
        log_alpha = log_l_prop - current_log_l

        if np.log(self._rng.uniform()) < log_alpha:
            return 1, theta_prop, log_l_prop, 1
        return 0, current_theta, current_log_l, 0

    def _trans_dimensional_death(self, current_theta: npt.NDArray[np.float64], current_log_l: float) -> Tuple[int, npt.NDArray[np.float64], float, int]:
        r"""
        Execute the 'Death' move: Transitioning from H1 -> H0.
        
        Preserves detailed balance by truncating the transient signal parameters 
        and evaluating the likelihood degradation against H0.
        """
        theta_prop = current_theta[:self._n_dim_h0]
        log_l_prop = self._log_l_h0(theta_prop)
        
        log_alpha = log_l_prop - current_log_l

        if np.log(self._rng.uniform()) < log_alpha:
            return 0, theta_prop, log_l_prop, 1
        return 1, current_theta, current_log_l, 0

    def execute(self, n_iterations: int = 100000, burn_in: int = 20000) -> Dict[str, Any]:
        """
        Execute the trans-dimensional Markov Chain Monte Carlo sequence.

        Parameters
        ----------
        n_iterations : int
            The total number of chain evaluations.
        burn_in : int
            The initial sequence of states to discard prior to reaching stationarity.

        Returns
        -------
        Dict[str, Any]
            The serialized traces of the model indicators and accepted parameter states.
        """
        model_indicator = np.zeros(n_iterations, dtype=np.int8)
        
        # Initialize chain at the center of the H0 prior
        current_model = 0
        current_theta = np.mean(self._bounds_h0, axis=1)
        current_log_l = self._log_l_h0(current_theta)

        jump_attempts = 0
        jump_accepts = 0

        # Output trace structures
        trace_h0: List[npt.NDArray[np.float64]] = []
        trace_h1: List[npt.NDArray[np.float64]] = []

        for i in range(n_iterations):
            move_type = self._rng.uniform()

            # 50% probability to propose an intra-model structural walk
            # 50% probability to propose a trans-dimensional state jump
            if move_type < 0.5:
                current_theta, current_log_l, _ = self._intra_model_move(current_model, current_theta, current_log_l)
            else:
                jump_attempts += 1
                if current_model == 0:
                    current_model, current_theta, current_log_l, acc = self._trans_dimensional_birth(current_theta, current_log_l)
                else:
                    current_model, current_theta, current_log_l, acc = self._trans_dimensional_death(current_theta, current_log_l)
                jump_accepts += acc

            model_indicator[i] = current_model
            
            if i >= burn_in:
                if current_model == 0:
                    trace_h0.append(current_theta.copy())
                else:
                    trace_h1.append(current_theta.copy())

        return {
            "model_indicator": model_indicator[burn_in:],
            "trace_h0": np.array(trace_h0),
            "trace_h1": np.array(trace_h1),
            "jump_acceptance_rate": jump_accepts / max(1, jump_attempts)
        }