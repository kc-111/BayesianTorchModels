"""Utility functions: freeze/unfreeze, sampling, entropy, parameter counting."""

import copy

import torch
import torch.nn as nn

from .module import Module
from .parameter import (
    AbstractParameter,
    DeterministicParameter,
    GaussianParameter,
    LaplacianParameter,
)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_stdvs(model: Module) -> None:
    """Freeze log_sigma / log_scale fields in-place (set requires_grad=False).

    After this, only means are trainable. Pass
    ``filter(lambda p: p.requires_grad, model.parameters())`` to the optimizer.

    Args:
        model: The model to freeze stdvs on.
    """
    for name, param in model.named_parameters():
        if "log_sigma" in name or "log_scale" in name:
            param.requires_grad_(False)


def freeze_means(model: Module) -> None:
    """Freeze mean fields in-place (set requires_grad=False).

    After this, only variational widths (log_sigma / log_scale) are trainable.

    Args:
        model: The model to freeze means on.
    """
    for name, param in model.named_parameters():
        if name.endswith(".mean"):
            param.requires_grad_(False)


def freeze_params(model: Module, names: list[str]) -> None:
    """Freeze specific named parameters (both mean and variational fields).

    Args:
        model: The model to freeze parameters on.
        names: Parameter names matching keys from ``model.get_parameters()``,
            e.g. ``["layers.0.W", "layers.1.b"]``.
    """
    param_names = set(names)
    for name, param in model.named_parameters():
        for pname in param_names:
            if pname in name:
                param.requires_grad_(False)
                break


def unfreeze_all(model: Module) -> None:
    """Restore requires_grad=True on all parameters.

    Args:
        model: The model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Parameter extraction (module-level convenience wrappers)
# ---------------------------------------------------------------------------

def flatten_means(model: Module) -> torch.Tensor:
    """Concatenate all parameter means into a flat vector.

    Args:
        model: The model to extract means from.

    Returns:
        1-D tensor of all concatenated means.
    """
    return model.flatten_means()


def flatten_stdvs(model: Module) -> torch.Tensor:
    """Concatenate all positive stdvs into a flat vector.

    Args:
        model: The model to extract stdvs from.

    Returns:
        1-D tensor of all concatenated stdvs (zeros for deterministic params).
    """
    return model.flatten_stdvs()


def get_parameter_count(model: Module) -> dict:
    """Count parameters by type.

    Args:
        model: The model to count parameters for.

    Returns:
        Dict with keys ``"total"``, ``"bayesian"``, ``"deterministic"``.
    """
    params = model.get_parameters()
    total = 0
    bayesian = 0
    deterministic = 0
    for p in params.values():
        n = p.mean.numel()
        total += n
        if isinstance(p, DeterministicParameter):
            deterministic += n
        else:
            bayesian += n
    return {"total": total, "bayesian": bayesian, "deterministic": deterministic}


def get_parameter_groups(model: Module) -> dict:
    """Group parameter names by distribution type.

    Args:
        model: The model to group parameters for.

    Returns:
        Dict with keys ``"gaussian"``, ``"laplacian"``, ``"deterministic"``,
        each mapping to a list of parameter name strings.
    """
    params = model.get_parameters()
    groups: dict[str, list[str]] = {
        "gaussian": [],
        "laplacian": [],
        "deterministic": [],
    }
    for name, p in params.items():
        if isinstance(p, GaussianParameter):
            groups["gaussian"].append(name)
        elif isinstance(p, LaplacianParameter):
            groups["laplacian"].append(name)
        else:
            groups["deterministic"].append(name)
    return groups


# ---------------------------------------------------------------------------
# Sampling & entropy
# ---------------------------------------------------------------------------

class _SampledParameterView:
    """Proxy that exposes a sampled value as .mean, mimicking AbstractParameter."""

    def __init__(self, sampled_value: torch.Tensor, original_param: AbstractParameter):
        self._sampled_value = sampled_value
        self._original = original_param

    @property
    def mean(self) -> torch.Tensor:
        return self._sampled_value

    @property
    def shape(self) -> tuple:
        return tuple(self._sampled_value.shape)

    def sample(self, generator=None) -> torch.Tensor:
        return self._sampled_value

    @property
    def __class__(self):
        """Allow isinstance checks to pass for the original parameter type."""
        return type(self._original)

    def __getattr__(self, name):
        return getattr(self._original, name)


class SampledModel:
    """Wrapper around a model with sampled parameter overrides.

    Stores reparameterized samples and uses ``torch.func.functional_call``
    to run the forward pass with overridden mean parameters. Gradients
    flow back through the samples to the original model's ``mean`` and
    ``log_sigma`` parameters.
    """

    def __init__(self, model: Module, param_overrides: dict[str, torch.Tensor]):
        self._model = model
        self._param_overrides = param_overrides
        # Build sampled parameter views for attribute access
        self._sampled_views: dict[str, _SampledParameterView] = {}
        bayesian_params = model.get_parameters()
        for name, sampled_val in param_overrides.items():
            # name is like "W.mean" or "layers.0.W.mean" — strip ".mean"
            param_name = name.rsplit(".mean", 1)[0]
            if param_name in bayesian_params:
                self._sampled_views[param_name] = _SampledParameterView(
                    sampled_val, bayesian_params[param_name]
                )

    def __call__(self, *args, **kwargs):
        """Forward pass using sampled parameters."""
        # Force sample=False since we already sampled
        kwargs["sample"] = False
        return torch.func.functional_call(
            self._model, self._param_overrides, args, kwargs
        )

    def get_parameters(self) -> dict[str, "_SampledParameterView"]:
        """Return sampled parameter views."""
        return dict(self._sampled_views)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying model.

        For sub-modules that contain sampled parameters, returns a proxy
        that exposes sampled values.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # Check if the model has this attribute
        model = object.__getattribute__(self, "_model")
        attr = getattr(model, name)
        if isinstance(attr, AbstractParameter):
            views = object.__getattribute__(self, "_sampled_views")
            if name in views:
                return views[name]
        return attr


def sample_all_parameters(
    model: Module, generator: torch.Generator | None = None
) -> SampledModel:
    """Return a SampledModel with all means replaced by reparameterized samples.

    The sampled values are computed via the reparameterization trick
    (``mean + exp(log_sigma) * noise``), so gradients flow back through
    the sample to both ``mean`` and ``log_sigma`` of the original model.
    The returned SampledModel can be called and will use the sampled means.

    Args:
        model: The model to sample parameters for.
        generator: Optional random number generator.

    Returns:
        A ``SampledModel`` whose parameter means are reparameterized
        samples. Useful for ODE integration or recurrent models where
        weights must be fixed across time-steps.
    """
    params = model.get_parameters()
    if not params:
        return SampledModel(model, {})

    param_overrides = {}
    for name, param in params.items():
        sampled_value = param.sample(generator)
        # Override the .mean parameter in the state dict
        param_overrides[name + ".mean"] = sampled_value

    return SampledModel(model, param_overrides)


def gaussian_entropy(model: Module) -> torch.Tensor:
    """Compute sum(log_sigma) over all Gaussian parameters.

    This is the variable part of the Gaussian entropy
    ``H(q) = 0.5 * d * (1 + log(2*pi)) + sum(log(sigma))``.
    Because sigma = exp(log_sigma), the log cancels the exp and the
    result is simply the sum of the stored ``log_sigma`` values.

    Args:
        model: The model to compute entropy for.

    Returns:
        Scalar sum of log_sigma across all Gaussian parameters.
    """
    params = model.get_parameters()
    total = torch.tensor(0.0)
    for p in params.values():
        if isinstance(p, GaussianParameter):
            total = total + torch.sum(p.log_sigma)
    return total


def laplacian_entropy(model: Module) -> torch.Tensor:
    """Compute sum(log_scale) over all Laplacian parameters.

    This is the variable part of the Laplace entropy
    ``H(q) = d * (1 + log(2)) + sum(log(b))``.
    Because scale = exp(log_scale), the result is simply the sum of
    the stored ``log_scale`` values.

    Args:
        model: The model to compute entropy for.

    Returns:
        Scalar sum of log_scale across all Laplacian parameters.
    """
    params = model.get_parameters()
    total = torch.tensor(0.0)
    for p in params.values():
        if isinstance(p, LaplacianParameter):
            total = total + torch.sum(p.log_scale)
    return total
