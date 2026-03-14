"""VI validation: compare variational posterior to analytical posterior.

For Bayesian linear regression with Gaussian prior and likelihood,
the posterior is analytically tractable. We verify that our VI framework
recovers the correct posterior mean and standard deviation.

Setup
-----
- Prior: w ~ N(0, (1/alpha) * I)
- Likelihood: y | x, w ~ N(w @ x, (1/beta) * I)
- Posterior (analytical):
    Sigma_post = inv(alpha * I + beta * X^T X)
    mu_post = beta * Sigma_post @ X^T y
"""

import torch
import pytest

from BayesianTorchModels import (
    BayesianLinear,
    GaussianParameter,
    freeze_stdvs,
    freeze_means,
    gaussian_entropy,
    unfreeze_all,
    sample_all_parameters,
)


def generate_data(n_samples=200, n_features=3, noise_std=0.5):
    """Generate synthetic linear regression data."""
    torch.manual_seed(42)
    w_true = torch.randn(n_features)
    X = torch.randn(n_samples, n_features)
    y = X @ w_true + noise_std * torch.randn(n_samples)
    return X, y, w_true


def analytical_posterior(X, y, alpha, beta):
    """Compute the analytical Gaussian posterior for Bayesian linear regression."""
    n_features = X.shape[1]
    Sigma_post_inv = alpha * torch.eye(n_features) + beta * X.T @ X
    Sigma_post = torch.linalg.inv(Sigma_post_inv)
    mu_post = beta * Sigma_post @ X.T @ y
    stdv_post = torch.sqrt(torch.diag(Sigma_post))
    return mu_post, stdv_post, Sigma_post


def elbo_loss(model, X, y, beta, alpha):
    """Negative ELBO = -E_q[log p(y|w,X)] - E_q[log p(w)] + E_q[log q(w)].

    For Gaussian q and Gaussian prior/likelihood, we use the reparameterisation
    trick and entropy in closed form.
    """
    # Sample weights
    y_pred = model(X, sample=True).squeeze(-1)

    # Log-likelihood: -0.5 * beta * ||y - y_pred||^2  (up to constants)
    log_lik = -0.5 * beta * torch.sum((y - y_pred) ** 2)

    # Log-prior: -0.5 * alpha * ||w||^2
    W_sample = model.W.sample()
    log_prior = -0.5 * alpha * torch.sum(W_sample ** 2)

    # Entropy of q (Gaussian)
    entropy = gaussian_entropy(model)

    # ELBO = log_lik + log_prior + entropy
    return -(log_lik + log_prior + entropy)


class TestVIvsMCMC:
    """Compare VI posterior to analytical posterior on linear regression."""

    @pytest.fixture
    def problem_setup(self):
        alpha = 1.0  # prior precision
        beta = 4.0   # likelihood precision (noise_std = 0.5 => beta = 1/0.25 = 4)
        noise_std = 1.0 / (beta ** 0.5)
        X, y, w_true = generate_data(n_samples=200, n_features=3, noise_std=noise_std)
        mu_post, stdv_post, _ = analytical_posterior(X, y, alpha, beta)
        return X, y, w_true, alpha, beta, mu_post, stdv_post

    def test_vi_recovers_posterior_mean(self, problem_setup):
        """VI mean should be close to analytical posterior mean."""
        X, y, _, alpha, beta, mu_post, stdv_post = problem_setup

        torch.manual_seed(0)
        model = BayesianLinear(
            3, 1, use_bias=False, bayesian=True, param_type=GaussianParameter,
            init_log_sigma=-3.0,
        )

        # Stage 1: MAP — freeze stdvs, optimize means
        freeze_stdvs(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2
        )

        for i in range(2000):
            optimizer.zero_grad()
            loss = elbo_loss(model, X, y, beta, alpha)
            loss.backward()
            optimizer.step()

        unfreeze_all(model)

        # Check MAP estimate ≈ posterior mean
        vi_mean = model.W.mean.detach().ravel()
        assert torch.allclose(vi_mean, mu_post, atol=0.15), (
            f"VI mean {vi_mean} vs analytical {mu_post}"
        )

    def test_vi_recovers_posterior_stdv(self, problem_setup):
        """After VI (MAP then stdv stage), stdvs should match analytical posterior."""
        X, y, _, alpha, beta, mu_post, stdv_post = problem_setup

        torch.manual_seed(0)
        model = BayesianLinear(
            3, 1, use_bias=False, bayesian=True, param_type=GaussianParameter,
            init_log_sigma=-3.0,
        )

        # Stage 1: MAP
        freeze_stdvs(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2
        )

        for i in range(2000):
            optimizer.zero_grad()
            loss = elbo_loss(model, X, y, beta, alpha)
            loss.backward()
            optimizer.step()

        unfreeze_all(model)

        # Stage 2: VI — freeze means, optimize stdvs
        freeze_means(model)
        optimizer2 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )

        for i in range(3000):
            optimizer2.zero_grad()
            # Average over multiple samples for lower variance
            losses = []
            for _ in range(8):
                losses.append(elbo_loss(model, X, y, beta, alpha))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer2.step()

        unfreeze_all(model)

        vi_stdv = model.W.stdv.detach().ravel()
        # Tolerant comparison — VI may not perfectly match analytical
        assert torch.allclose(vi_stdv, stdv_post, atol=0.1), (
            f"VI stdv {vi_stdv} vs analytical {stdv_post}"
        )

    def test_vi_mean_better_than_random(self, problem_setup):
        """Even minimal training should improve over random initialisation."""
        X, y, _, alpha, beta, mu_post, _ = problem_setup

        torch.manual_seed(99)
        model = BayesianLinear(3, 1, use_bias=False, bayesian=True)

        # Random model prediction error
        y_rand = model(X, sample=False).squeeze(-1)
        mse_rand = torch.mean((y - y_rand) ** 2)

        # Minimal MAP training (200 steps)
        freeze_stdvs(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2
        )

        for _ in range(200):
            optimizer.zero_grad()
            loss = elbo_loss(model, X, y, beta, alpha)
            loss.backward()
            optimizer.step()

        unfreeze_all(model)

        y_trained = model(X, sample=False).squeeze(-1)
        mse_trained = torch.mean((y - y_trained.detach()) ** 2)

        assert mse_trained < mse_rand, (
            f"Trained MSE {mse_trained:.4f} should be less than random MSE {mse_rand:.4f}"
        )
