"""Base Module class with parameter introspection utilities."""

import torch
import torch.nn as nn

from .parameter import AbstractParameter, DeterministicParameter, GaussianParameter


class Module(nn.Module):
    """Base class for Bayesian models.

    Provides recursive parameter introspection: ``get_parameters``,
    ``flatten_means``, ``flatten_stdvs``, and ``flatten_log_sigmas``.
    """

    def get_parameters(self) -> dict[str, AbstractParameter]:
        """Return a flat dict mapping dotted names to parameter leaves.

        Returns:
            Dict like ``{"W": GaussianParameter(...), "b": DeterministicParameter(...)}``.
            Nested modules produce dotted names (``"layer1.W"``); ModuleLists
            produce indexed names (``"layers.0.W"``).
        """
        result: dict[str, AbstractParameter] = {}
        for name, module in self.named_modules():
            if isinstance(module, AbstractParameter) and module is not self:
                result[name] = module
        return result

    def flatten_means(self) -> torch.Tensor:
        """Concatenate all parameter means into a single flat vector.

        Returns:
            1-D tensor of length ``sum(p.mean.numel() for p in parameters)``.
        """
        params = self.get_parameters()
        if not params:
            return torch.tensor([])
        return torch.cat([p.mean.detach().ravel() for p in params.values()])

    def flatten_stdvs(self) -> torch.Tensor:
        """Concatenate all positive stdvs into a flat vector.

        Deterministic parameters contribute zeros.

        Returns:
            1-D tensor with the same length as ``flatten_means()``.
        """
        params = self.get_parameters()
        if not params:
            return torch.tensor([])
        parts = []
        for p in params.values():
            if isinstance(p, GaussianParameter):
                parts.append(p.stdv.detach().ravel())
            elif hasattr(p, "scale"):
                parts.append(p.scale.detach().ravel())
            else:
                parts.append(torch.zeros(p.mean.numel()))
        return torch.cat(parts)

    def flatten_log_sigmas(self) -> torch.Tensor:
        """Concatenate all unconstrained log-scale params into a flat vector.

        These are the values the optimiser updates directly
        (``log_sigma`` for Gaussian, ``log_scale`` for Laplacian).
        Deterministic parameters contribute zeros.

        Returns:
            1-D tensor with the same length as ``flatten_means()``.
        """
        params = self.get_parameters()
        if not params:
            return torch.tensor([])
        parts = []
        for p in params.values():
            if hasattr(p, "log_sigma"):
                parts.append(p.log_sigma.detach().ravel())
            elif hasattr(p, "log_scale"):
                parts.append(p.log_scale.detach().ravel())
            else:
                parts.append(torch.zeros(p.mean.numel()))
        return torch.cat(parts)
