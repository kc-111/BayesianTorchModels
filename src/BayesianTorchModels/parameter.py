"""Bayesian parameter classes built on torch.nn.

Each parameter stores a mean and an unconstrained log-scale parameter.
Positivity of sigma / scale is guaranteed via the exp transform.
Sampling uses reparameterization tricks for gradient flow.
"""

from abc import abstractmethod
from typing import Type

import torch
import torch.nn as nn


class AbstractParameter(nn.Module):
    """Base class for all parameters."""

    def __init__(self, mean: torch.Tensor):
        super().__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean, dtype=torch.float32))

    @abstractmethod
    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a sample via the reparameterization trick.

        Args:
            generator: Optional random number generator.

        Returns:
            Sampled tensor with the same shape as ``mean``.
        """
        ...

    @property
    def shape(self) -> tuple:
        """Shape of the parameter."""
        return tuple(self.mean.shape)


class DeterministicParameter(AbstractParameter):
    """Fixed parameter (no variational distribution)."""

    def __init__(self, mean: torch.Tensor):
        super().__init__(mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return the mean (no randomness).

        Args:
            generator: Optional random number generator (ignored).

        Returns:
            The stored mean value.
        """
        return self.mean


class GaussianParameter(AbstractParameter):
    """Gaussian variational parameter with reparameterization.

    The actual standard deviation is ``exp(log_sigma)``, which guarantees
    positivity via the exponential transform.  The unconstrained
    ``log_sigma`` is the quantity the optimiser updates directly.
    """

    def __init__(self, mean: torch.Tensor, log_sigma: torch.Tensor):
        super().__init__(mean)
        self.log_sigma = nn.Parameter(torch.as_tensor(log_sigma, dtype=torch.float32))

    @property
    def stdv(self) -> torch.Tensor:
        """Positive standard deviation: ``exp(log_sigma)``."""
        return torch.exp(self.log_sigma)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample via ``mean + exp(log_sigma) * N(0, 1)``.

        Args:
            generator: Optional random number generator.

        Returns:
            Sampled tensor with the same shape as ``mean``.
        """
        noise = torch.randn(self.mean.shape, dtype=self.mean.dtype,
                            device=self.mean.device, generator=generator)
        return self.mean + self.stdv * noise


class LaplacianParameter(AbstractParameter):
    """Laplace variational parameter using inverse-CDF reparameterization.

    The actual scale is ``exp(log_scale)``, guaranteed positive via the
    exponential transform.
    """

    def __init__(self, mean: torch.Tensor, log_scale: torch.Tensor):
        super().__init__(mean)
        self.log_scale = nn.Parameter(torch.as_tensor(log_scale, dtype=torch.float32))

    @property
    def scale(self) -> torch.Tensor:
        """Positive scale: ``exp(log_scale)``."""
        return torch.exp(self.log_scale)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample via the inverse-CDF Laplace reparameterization.

        Args:
            generator: Optional random number generator.

        Returns:
            Sampled tensor with the same shape as ``mean``.
        """
        u = torch.empty(self.mean.shape, dtype=self.mean.dtype,
                        device=self.mean.device).uniform_(1e-7, 1 - 1e-7,
                                                          generator=generator)
        return self.mean - self.scale * torch.sign(u - 0.5) * torch.log1p(
            -2 * torch.abs(u - 0.5)
        )


def make_parameter(
    value: torch.Tensor,
    *,
    bayesian: bool = True,
    param_type: Type[AbstractParameter] = GaussianParameter,
    init_log_sigma: float = -5.0,
) -> AbstractParameter:
    """Wrap an arbitrary tensor as a Bayesian or deterministic parameter.

    Args:
        value: Initial mean value. Any shape (scalar, vector, matrix, tensor).
        bayesian: If True (default), wrap in ``param_type``. If False, wrap
            in ``DeterministicParameter``.
        param_type: ``GaussianParameter`` (default) or ``LaplacianParameter``.
        init_log_sigma: Initial value for the unconstrained log-scale field
            (``log_sigma`` or ``log_scale``).  The effective initial standard
            deviation / scale is ``exp(init_log_sigma)``.

    Returns:
        An ``AbstractParameter`` instance wrapping ``value``.
    """
    value = torch.as_tensor(value, dtype=torch.float32)
    if not bayesian:
        return DeterministicParameter(mean=value)
    if param_type is GaussianParameter:
        return GaussianParameter(
            mean=value,
            log_sigma=torch.full_like(value, init_log_sigma),
        )
    if param_type is LaplacianParameter:
        return LaplacianParameter(
            mean=value,
            log_scale=torch.full_like(value, init_log_sigma),
        )
    raise ValueError(f"Unknown param_type: {param_type}")
