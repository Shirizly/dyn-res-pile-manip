"""
Action sampling strategies for MPC candidate generation.

Different sampling strategies (random, learned priors, importance sampling, etc.)
can be implemented as subclasses of ActionSampler and plugged into run_simple_mpc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import torch


class ActionSampler(ABC):
    """
    Base class for MPC action sampling strategies.

    Each MPC step samples n_sample candidate action sequences of length n_ahead.
    Different samplers can implement different strategies (random, learned,
    importance-weighted, etc.).
    """

    @abstractmethod
    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate a batch of action sequences.

        Parameters
        ----------
        n_sample : int
            Number of independent candidates to sample
        n_ahead : int
            Planning horizon (rollout length)
        act_lo : np.ndarray (4,)
            Lower bounds for actions [sx, sy, ex, ey]
        act_hi : np.ndarray (4,)
            Upper bounds for actions [sx, sy, ex, ey]
        device : str
            PyTorch device ('cuda' or 'cpu')

        Returns
        -------
        act_seqs : torch.Tensor (n_sample, n_ahead, 4)
            Sampled action sequences, requires_grad=True for optimization
        """
        pass


class RandomUniformSampler(ActionSampler):
    """
    Sample actions uniformly at random from the workspace bounds.

    This is the default / baseline strategy: each action component is drawn
    independently from the specified bounds. Fast and simple.
    """

    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate uniformly random action sequences.

        Each of the n_sample candidates independently samples each action
        component uniformly from [act_lo, act_hi].
        """
        act_np = np.random.uniform(
            act_lo, act_hi, (n_sample, n_ahead, 4)
        ).astype(np.float32)
        act_seqs = torch.tensor(act_np, device=device, requires_grad=True)
        return act_seqs


# Factory function for easy selection
def make_action_sampler(sampler_type: str = 'uniform') -> ActionSampler:
    """
    Create an action sampler by name.

    Parameters
    ----------
    sampler_type : str
        Name of the sampler: 'uniform' (default), or custom registered type

    Returns
    -------
    sampler : ActionSampler
        Instantiated sampler ready for use

    Examples
    --------
    >>> sampler = make_action_sampler('uniform')
    >>> acts = sampler.sample(n_sample=512, n_ahead=1, ...)
    """
    samplers = {
        'uniform': RandomUniformSampler,
    }

    if sampler_type not in samplers:
        raise ValueError(
            f"Unknown sampler '{sampler_type}'. "
            f"Available: {sorted(samplers.keys())}"
        )

    return samplers[sampler_type]()
