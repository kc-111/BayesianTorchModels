"""ODE integration example using BayesianTorchModels + torchdiffeq.

Demonstrates the 'sample once, integrate deterministically' pattern:
1. Define a Bayesian ODE function (neural ODE right-hand-side)
2. Use sample_all_parameters to fix weights for one trajectory
3. Integrate with torchdiffeq
4. Loop over generators for posterior trajectory samples
"""

import torch
import pytest

from BayesianTorchModels import (
    BayesianLinear,
    Module,
    sample_all_parameters,
)

torchdiffeq = pytest.importorskip("torchdiffeq")


class BayesianODEFunc(Module):
    """Bayesian neural ODE right-hand side: dx/dt = NN(x).

    NOT part of the library — just a user-defined model for testing.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = BayesianLinear(state_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, state_dim)

    def forward(self, x: torch.Tensor, *, sample: bool = False,
                generator: torch.Generator | None = None) -> torch.Tensor:
        h = torch.tanh(self.layer1(x, sample=sample, generator=generator))
        return self.layer2(h, sample=sample, generator=generator)


def integrate_ode(func, y0, t0, t1, saveat_ts):
    """Integrate an ODE using torchdiffeq with dopri5 solver."""

    def vector_field(t, y):
        return func(y, sample=False)

    sol = torchdiffeq.odeint(vector_field, y0, saveat_ts, method="dopri5")
    return sol


class TestBayesianODE:
    """Tests demonstrating Bayesian ODE usage patterns."""

    def test_single_trajectory(self):
        """A single sampled model should produce a deterministic trajectory."""
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)

        # Sample weights once
        g = torch.Generator().manual_seed(1)
        sampled_func = sample_all_parameters(func, g)

        # Means should differ from the original (they are samples now)
        assert not torch.allclose(
            sampled_func.get_parameters()["layer1.W"].mean, func.layer1.W.mean
        )

        # Integrate
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 20)
        traj = integrate_ode(sampled_func, y0, t0=0.0, t1=1.0, saveat_ts=ts)

        assert traj.shape == (20, 2)
        assert torch.all(torch.isfinite(traj))

    def test_trajectory_reproducibility(self):
        """Same generator seed should produce identical trajectories."""
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 10)

        g1 = torch.Generator().manual_seed(42)
        s1 = sample_all_parameters(func, g1)
        t1 = integrate_ode(s1, y0, 0.0, 1.0, ts)

        g2 = torch.Generator().manual_seed(42)
        s2 = sample_all_parameters(func, g2)
        t2 = integrate_ode(s2, y0, 0.0, 1.0, ts)

        assert torch.allclose(t1, t2)

    def test_different_generators_different_trajectories(self):
        """Different generator seeds should yield different trajectories."""
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 10)

        g1 = torch.Generator().manual_seed(1)
        s1 = sample_all_parameters(func, g1)
        t1 = integrate_ode(s1, y0, 0.0, 1.0, ts)

        g2 = torch.Generator().manual_seed(2)
        s2 = sample_all_parameters(func, g2)
        t2 = integrate_ode(s2, y0, 0.0, 1.0, ts)

        assert not torch.allclose(t1, t2)

    def test_posterior_samples_via_loop(self):
        """Loop over generator seeds to get posterior trajectory samples."""
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 10)
        n_samples = 5

        trajectories = []
        for i in range(n_samples):
            g = torch.Generator().manual_seed(i)
            sampled = sample_all_parameters(func, g)
            traj = integrate_ode(sampled, y0, 0.0, 1.0, ts)
            trajectories.append(traj)

        trajectories = torch.stack(trajectories)
        assert trajectories.shape == (n_samples, 10, 2)
        assert torch.all(torch.isfinite(trajectories))

        # Trajectories should have non-zero variance across samples
        traj_std = trajectories.std(dim=0)
        assert torch.any(traj_std > 1e-6)

    def test_ode_deterministic_forward(self):
        """Using means directly (no sampling) should work and be consistent."""
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 10)

        t1 = integrate_ode(func, y0, 0.0, 1.0, ts)
        t2 = integrate_ode(func, y0, 0.0, 1.0, ts)

        assert torch.allclose(t1, t2)
        assert t1.shape == (10, 2)

    def test_gradient_through_ode(self):
        """Gradients should flow through sample -> ODE integrate -> loss.

        This is the actual training pipeline: we sample weights via the
        reparameterisation trick, integrate the ODE with those fixed
        weights, compute a loss, and differentiate back to both mean and
        log_sigma of the original model.
        """
        torch.manual_seed(0)
        func = BayesianODEFunc(state_dim=2, hidden_dim=8)
        y0 = torch.tensor([1.0, 0.0])
        ts = torch.linspace(0, 1, 5)
        target = torch.zeros(5, 2)

        sampled = sample_all_parameters(func)
        traj = integrate_ode(sampled, y0, 0.0, 1.0, ts)
        loss = torch.mean((traj - target) ** 2)

        assert torch.isfinite(loss)
        loss.backward()

        # Grads on means should be non-zero
        mean_grads = [
            p.grad for name, p in func.named_parameters()
            if "mean" in name and p.grad is not None
        ]
        assert any(torch.any(g != 0) for g in mean_grads), \
            "Expected non-zero gradients on means"

        # Grads on log_sigma should also be non-zero (reparameterisation trick)
        stdv_grads = [
            p.grad for name, p in func.named_parameters()
            if "log_sigma" in name and p.grad is not None
        ]
        assert any(torch.any(g != 0) for g in stdv_grads), \
            "Expected non-zero gradients on log_sigma (reparameterisation)"
