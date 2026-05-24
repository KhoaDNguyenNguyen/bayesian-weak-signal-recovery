import pickle
import numpy as np
import numpy.typing as npt
import dynesty
from typing import Callable, Dict, Any, Optional
from pathlib import Path


class NestedInferenceEngine:
    """
    Execution engine for Dynamic Nested Sampling to evaluate the marginal 
    likelihood (Bayesian Evidence) and compute posterior probability distributions.

    Nested Sampling computes the multi-dimensional integral:
        Z = \int \mathcal{L}(\theta) \pi(\theta) d\theta
    by transforming the volume element to prior mass X, yielding:
        Z = \int_0^1 \mathcal{L}(X) dX
    where \mathcal{L} is the likelihood function and \pi is the prior density.
    """

    def __init__(
        self,
        log_likelihood: Callable[[npt.NDArray[np.float64]], float],
        prior_transform: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        n_dim: int,
        n_live_points: int = 1000,
        bounding_method: str = 'multi',
        sampling_method: str = 'auto',
        pool: Optional[Any] = None,
        queue_size: Optional[int] = None
    ) -> None:
        """
        Initialize the Nested Sampling execution engine.

        Parameters
        ----------
        log_likelihood : Callable[[npt.NDArray[np.float64]], float]
            The log-likelihood function formulation.
        prior_transform : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            The prior transformation function mapping the unit hypercube to 
            the physical parameter space.
        n_dim : int
            Dimensionality of the parameter space (D).
        n_live_points : int, optional
            Number of live points maintained during the sampling process. 
            Default is 1000.
        bounding_method : str, optional
            Method used to bound the prior volume. Default is 'multi'.
        sampling_method : str, optional
            Method used to sample the bounded prior volume. Default is 'auto'.
        pool : Optional[Any], optional
            A multiprocessing execution pool for parallel likelihood evaluation.
        queue_size : Optional[int], optional
            The number of parallel evaluations to queue.

        Raises
        ------
        ValueError
            If the number of dimensions or live points is non-positive.
        """
        if n_dim <= 0:
            raise ValueError("Dimensionality of the parameter space must be strictly positive.")
        if n_live_points <= 0:
            raise ValueError("The number of live points must be strictly positive.")

        self._n_dim = n_dim
        self._n_live_points = n_live_points
        self._log_likelihood = log_likelihood
        self._prior_transform = prior_transform
        
        self._sampler = dynesty.NestedSampler(
            loglikelihood=self._log_likelihood,
            prior_transform=self._prior_transform,
            ndim=self._n_dim,
            nlive=self._n_live_points,
            bound=bounding_method,
            sample=sampling_method,
            pool=pool,
            queue_size=queue_size
        )
        self._results: Optional[dynesty.results.Results] = None

    def execute(self, dlogz: float = 0.1) -> Dict[str, Any]:
        """
        Execute the nested sampling algorithm until the convergence criterion is met.

        Parameters
        ----------
        dlogz : float, optional
            The convergence tolerance representing the remaining log-evidence 
            contribution (\Delta \ln Z). Default is 0.1.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the extracted posterior samples, 
            importance weights, and the global Bayesian evidence.

        Raises
        ------
        ValueError
            If the convergence tolerance is negative.
        RuntimeError
            If the sampling algorithm encounters a numerical instability.
        """
        if dlogz < 0.0:
            raise ValueError("Convergence tolerance dlogz must be non-negative.")

        try:
            self._sampler.run_nested(dlogz=dlogz)
        except Exception as e:
            raise RuntimeError(f"Nested sampling execution failed: {e}")

        self._results = self._sampler.results

        return {
            'samples': self._results.samples,
            'weights': np.exp(self._results.logwt - self._results.logz[-1]),
            'log_evidence': self._results.logz[-1],
            'log_evidence_err': self._results.logzerr[-1],
            'log_likelihoods': self._results.logl
        }

    def serialize_results(self, output_path: Path) -> None:
        """
        Serialize the complete nested sampling results to a binary file.

        Parameters
        ----------
        output_path : Path
            The file system path where the binary object will be stored.
        """
        if self._results is None:
            raise RuntimeError("Cannot serialize results: Nested sampling has not been executed.")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as file_descriptor:
                pickle.dump(self._results, file_descriptor, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError as e:
            raise IOError(f"Failed to serialize inference results to {output_path}: {e}")