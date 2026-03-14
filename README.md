# BayesianTorchModels

Modular Bayesian model framework in PyTorch.

Port of [BayesianJaxModels](https://github.com/kc-111/BayesianJaxModels) from JAX/Equinox to PyTorch.

Provides parameter types with reparameterized sampling, a `Module` base class with
parameter introspection, Bayesian layers, and utility functions for freeze/unfreeze,
sampling, and entropy computation. You own the training loop — this package provides
only the differentiable model and utilities.

## Installation

```bash
pip install git+https://github.com/kc-111/BayesianTorchModels.git
```

With test dependencies:

```bash
pip install "BayesianTorchModels[test] @ git+https://github.com/kc-111/BayesianTorchModels.git"
```

## Dependencies

- `torch`
- `torchdiffeq` (optional, for ODE integration)

## Quick start

```python
import torch
from BayesianTorchModels import (
    BayesianLinear, Module, make_parameter,
    freeze_stdvs, freeze_means, unfreeze_all,
    sample_all_parameters, gaussian_entropy,
)
```

### Defining a model

Subclass `Module` and compose with `BayesianLinear` or `make_parameter`:

```python
import torch.nn as nn

class MLP(Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesianLinear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

    def forward(self, x, *, sample=True, generator=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample, generator=generator)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

model = MLP([4, 32, 1])
```

### Forward pass

```python
# Stochastic (samples from variational distribution):
y = model(x, sample=True)

# Deterministic (uses means only):
y = model(x, sample=False)

# With explicit generator for reproducibility:
g = torch.Generator().manual_seed(42)
y = model(x, sample=True, generator=g)
```

### Two-stage VI training

**Stage 1 — MAP (optimize means, freeze stdvs):**

```python
freeze_stdvs(model)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

for i in range(1000):
    optimizer.zero_grad()
    y_pred = model(x_train, sample=True)
    loss = torch.mean((y_train - y_pred) ** 2)
    loss.backward()
    optimizer.step()

unfreeze_all(model)
```

**Stage 2 — VI (optimize stdvs, freeze means):**

```python
freeze_means(model)
optimizer2 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

for i in range(1000):
    optimizer2.zero_grad()
    y_pred = model(x_train, sample=True)
    nll = 0.5 * beta * torch.sum((y_train - y_pred) ** 2)
    loss = nll - gaussian_entropy(model)
    loss.backward()
    optimizer2.step()

unfreeze_all(model)
```

### ODE integration pattern

For models where parameters must be fixed across time steps (ODEs, RNNs),
sample once then integrate deterministically:

```python
import torchdiffeq

sampled = sample_all_parameters(model)  # sample weights once

def vector_field(t, y):
    return sampled(y)  # fixed weights, sample=False automatic

sol = torchdiffeq.odeint(vector_field, y0, t_span)
loss = torch.mean((sol - target) ** 2)
loss.backward()
# grads contain derivatives w.r.t. both mean AND log_sigma
```

### Introspection

```python
params = model.get_parameters()      # {"layers.0.W": GaussianParameter, ...}
means  = model.flatten_means()       # 1-D tensor of all means
stdvs  = model.flatten_stdvs()       # 1-D tensor of all stdvs (0 for deterministic)
counts = get_parameter_count(model)  # {"total": N, "bayesian": M, "deterministic": K}
groups = get_parameter_groups(model) # {"gaussian": [...], "laplacian": [...], "deterministic": [...]}
```

## API differences from JAX version

| JAX API | PyTorch API |
|---------|-------------|
| `model(x, key=key, sample=True)` | `model(x, sample=True, generator=gen)` |
| `freeze_stdvs(model)` → `(dyn, static)` | `freeze_stdvs(model)` → in-place mutation |
| `unfreeze_all(dyn, static)` → model | `unfreeze_all(model)` → in-place mutation |
| `sample_all_parameters(model, rng)` → model copy | `sample_all_parameters(model, gen)` → `SampledModel` |
| `layers[0].W` param names | `layers.0.W` param names |
| `jax.vmap`, `jax.jit` | loop / `torch.compile` (optional) |

## Running tests

```bash
pip install BayesianTorchModels[test]
pytest tests/ -v
```

## License

MIT
