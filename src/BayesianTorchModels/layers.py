"""Bayesian layers built on the Module base class."""

import math
from typing import Type

import torch
import torch.nn as nn

from .module import Module
from .parameter import (
    AbstractParameter,
    GaussianParameter,
    make_parameter,
)


class BayesianLinear(Module):
    """Bayesian fully-connected layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        use_bias: Whether to include a bias term.
        bayesian: If True (default), weights use a variational distribution.
        bayesian_bias: If True (default), bias also uses a variational
            distribution.
        param_type: ``GaussianParameter`` (default) or ``LaplacianParameter``.
        init_log_sigma: Initial value for the unconstrained log-scale
            parameter.  Effective initial stdv is ``exp(init_log_sigma)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        bayesian: bool = True,
        bayesian_bias: bool = True,
        param_type: Type[AbstractParameter] = GaussianParameter,
        init_log_sigma: float = -5.0,
    ):
        super().__init__()
        # Xavier uniform initialisation
        limit = math.sqrt(6.0 / (in_features + out_features))
        W_init = torch.empty(out_features, in_features).uniform_(-limit, limit)
        self.W = make_parameter(
            W_init, bayesian=bayesian, param_type=param_type,
            init_log_sigma=init_log_sigma,
        )

        if use_bias:
            b_init = torch.zeros(out_features)
            self.b = make_parameter(
                b_init, bayesian=bayesian_bias, param_type=param_type,
                init_log_sigma=init_log_sigma,
            )
        else:
            self.b = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., in_features)``.
            sample: If True, sample weights from the variational distribution.
                If False, use the means (MAP / deterministic mode).
            generator: Optional random number generator. Used when ``sample=True``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        if sample:
            W = self.W.sample(generator)
        else:
            W = self.W.mean

        out = x @ W.T

        if self.b is not None:
            if sample:
                b = self.b.sample(generator)
            else:
                b = self.b.mean
            out = out + b

        return out
