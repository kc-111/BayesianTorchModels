"""Comprehensive unit tests for BayesianTorchModels."""

import torch
import torch.nn as nn
import pytest

from BayesianTorchModels import (
    AbstractParameter,
    BayesianLinear,
    DeterministicParameter,
    GaussianParameter,
    LaplacianParameter,
    Module,
    SampledModel,
    flatten_means,
    flatten_stdvs,
    freeze_means,
    freeze_stdvs,
    gaussian_entropy,
    get_parameter_count,
    get_parameter_groups,
    laplacian_entropy,
    make_parameter,
    sample_all_parameters,
    unfreeze_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(Module):
    """Two-layer MLP for testing nested module introspection."""

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


class MixedModel(Module):
    """Model mixing Gaussian, Laplacian, and Deterministic parameters."""

    def __init__(self):
        super().__init__()
        self.gauss_layer = BayesianLinear(4, 3, param_type=GaussianParameter)
        self.laplace_layer = BayesianLinear(3, 2, param_type=LaplacianParameter)
        self.det_layer = BayesianLinear(2, 1, bayesian=False)

    def forward(self, x, *, sample=True, generator=None):
        x = torch.relu(self.gauss_layer(x, sample=sample, generator=generator))
        x = torch.relu(self.laplace_layer(x, sample=sample, generator=generator))
        x = self.det_layer(x, sample=sample, generator=generator)
        return x


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------


class TestParameters:
    def test_deterministic_sample_returns_mean(self):
        p = DeterministicParameter(mean=torch.tensor([1.0, 2.0]))
        assert torch.allclose(p.sample(), p.mean)

    def test_gaussian_sample_shape(self):
        p = GaussianParameter(
            mean=torch.zeros(3, 4), log_sigma=torch.ones(3, 4) * (-2.0)
        )
        s = p.sample()
        assert s.shape == (3, 4)

    def test_gaussian_stdv_positive(self):
        p = GaussianParameter(
            mean=torch.zeros(5), log_sigma=torch.tensor([-5.0, -1.0, 0.0, 1.0, 2.0])
        )
        assert torch.all(p.stdv > 0)

    def test_gaussian_sample_distribution(self):
        """Many samples should have mean ~ param.mean and std ~ param.stdv."""
        mean = torch.tensor([1.0, -2.0])
        log_sigma = torch.tensor([-1.0, 0.0])  # stdv = [~0.368, 1.0]
        p = GaussianParameter(mean=mean, log_sigma=log_sigma)
        torch.manual_seed(0)
        samples = torch.stack([p.sample() for _ in range(10000)])
        assert torch.allclose(samples.mean(dim=0), mean, atol=0.05)
        assert torch.allclose(samples.std(dim=0), p.stdv.detach(), atol=0.05)

    def test_laplacian_sample_shape(self):
        p = LaplacianParameter(
            mean=torch.zeros(5), log_scale=torch.ones(5) * (-2.0)
        )
        s = p.sample()
        assert s.shape == (5,)

    def test_laplacian_scale_positive(self):
        p = LaplacianParameter(
            mean=torch.zeros(3), log_scale=torch.tensor([-5.0, 0.0, 1.0])
        )
        assert torch.all(p.scale > 0)

    def test_laplacian_sample_distribution(self):
        """Laplacian samples should have mean ~ param.mean."""
        mean = torch.tensor([3.0])
        p = LaplacianParameter(mean=mean, log_scale=torch.tensor([-0.5]))
        torch.manual_seed(0)
        samples = torch.stack([p.sample() for _ in range(10000)])
        assert torch.allclose(samples.mean(dim=0), mean, atol=0.1)

    def test_parameter_shape_property(self):
        p = GaussianParameter(mean=torch.zeros(2, 3), log_sigma=torch.zeros(2, 3))
        assert p.shape == (2, 3)


# ---------------------------------------------------------------------------
# BayesianLinear tests
# ---------------------------------------------------------------------------


class TestBayesianLinear:
    def test_forward_sample_true(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        y = layer(x, sample=True)
        assert y.shape == (2, 3)

    def test_forward_sample_false(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        y = layer(x, sample=False)
        assert y.shape == (2, 3)

    def test_deterministic_mode(self):
        """sample=False should give identical outputs each call."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        y1 = layer(x, sample=False)
        y2 = layer(x, sample=False)
        assert torch.allclose(y1, y2)

    def test_stochastic_mode_varies(self):
        """Different calls with sample=True should give different outputs."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        y1 = layer(x, sample=True)
        y2 = layer(x, sample=True)
        assert not torch.allclose(y1, y2)

    def test_no_bias(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, use_bias=False)
        assert layer.b is None
        x = torch.ones(2, 4)
        y = layer(x, sample=True)
        assert y.shape == (2, 3)

    def test_bayesian_bias_default(self):
        """bayesian_bias=True is the default."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        assert isinstance(layer.b, GaussianParameter)

    def test_non_bayesian(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False)
        assert isinstance(layer.W, DeterministicParameter)
        assert isinstance(layer.b, DeterministicParameter)

    def test_laplacian_param_type(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, param_type=LaplacianParameter)
        assert isinstance(layer.W, LaplacianParameter)

    def test_generator_reproducibility(self):
        """Same generator seed should produce identical outputs."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        g1 = torch.Generator().manual_seed(42)
        y1 = layer(x, sample=True, generator=g1)
        g2 = torch.Generator().manual_seed(42)
        y2 = layer(x, sample=True, generator=g2)
        assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Module introspection tests
# ---------------------------------------------------------------------------


class TestModuleIntrospection:
    def test_get_parameters_single_layer(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        params = layer.get_parameters()
        assert "W" in params
        assert "b" in params
        assert isinstance(params["W"], GaussianParameter)
        assert isinstance(params["b"], GaussianParameter)

    def test_get_parameters_deterministic_bias(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian_bias=False)
        params = layer.get_parameters()
        assert isinstance(params["b"], DeterministicParameter)

    def test_get_parameters_mlp(self):
        torch.manual_seed(0)
        mlp = SimpleMLP([4, 8, 3])
        params = mlp.get_parameters()
        assert "layers.0.W" in params
        assert "layers.0.b" in params
        assert "layers.1.W" in params
        assert "layers.1.b" in params
        assert len(params) == 4

    def test_flatten_means_shape(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        means = layer.flatten_means()
        # W: 3*4 = 12, b: 3 => total 15
        assert means.shape == (15,)

    def test_flatten_stdvs_shape(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        stdvs = layer.flatten_stdvs()
        # W: 12 stdvs, b: 3 stdvs (both bayesian by default) => 15
        assert stdvs.shape == (15,)
        assert torch.all(stdvs > 0)

    def test_flatten_stdvs_deterministic_bias(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian_bias=False)
        stdvs = layer.flatten_stdvs()
        assert stdvs.shape == (15,)
        assert torch.all(stdvs[:12] > 0)
        assert torch.allclose(stdvs[12:], torch.tensor(0.0))

    def test_flatten_log_sigmas_shape(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        raw = layer.flatten_log_sigmas()
        assert raw.shape == (15,)

    def test_mlp_forward(self):
        torch.manual_seed(0)
        mlp = SimpleMLP([4, 8, 3])
        x = torch.ones(2, 4)
        y = mlp(x, sample=True)
        assert y.shape == (2, 3)

    def test_mlp_forward_deterministic(self):
        torch.manual_seed(0)
        mlp = SimpleMLP([4, 8, 3])
        x = torch.ones(2, 4)
        y = mlp(x, sample=False)
        assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# Freeze / unfreeze tests
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    def test_freeze_stdvs_grad_on_means_only(self):
        """After freezing stdvs, gradients should only flow through means."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        freeze_stdvs(layer)

        x = torch.ones(1, 4)
        y = layer(x, sample=True)
        loss = y.sum()
        loss.backward()

        # Means should have gradients
        assert layer.W.mean.grad is not None
        assert torch.any(layer.W.mean.grad != 0)

        # log_sigma should not have gradients (frozen)
        assert layer.W.log_sigma.grad is None
        assert not layer.W.log_sigma.requires_grad

        # Cleanup
        unfreeze_all(layer)

    def test_freeze_means_grad_on_stdvs_only(self):
        """After freezing means, gradients should only flow through stdvs."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian_bias=True)
        freeze_means(layer)

        x = torch.ones(1, 4)
        y = layer(x, sample=True)
        loss = y.sum()
        loss.backward()

        # Means should be frozen
        assert not layer.W.mean.requires_grad
        assert layer.W.mean.grad is None

        # log_sigma should have gradients
        assert layer.W.log_sigma.grad is not None

        # Cleanup
        unfreeze_all(layer)

    def test_unfreeze_all_roundtrip(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)
        y_orig = layer(x, sample=False)

        freeze_stdvs(layer)
        unfreeze_all(layer)
        y_restored = layer(x, sample=False)

        assert torch.allclose(y_orig, y_restored)

    def test_freeze_stdvs_optimizer_pattern(self):
        """Verify the optimizer pattern: only trainable params passed."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        freeze_stdvs(layer)

        trainable = list(filter(lambda p: p.requires_grad, layer.parameters()))
        # Only means should be trainable (W.mean and b.mean)
        assert len(trainable) == 2

        unfreeze_all(layer)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_sample_all_parameters(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        sampled = sample_all_parameters(layer)

        # Means should differ from original (they are samples now)
        params = sampled.get_parameters()
        assert not torch.allclose(params["W"].mean, layer.W.mean)

        # Parameter types are preserved
        assert isinstance(params["W"], GaussianParameter)

        # Forward pass should work
        x = torch.ones(2, 4)
        y = sampled(x)
        assert y.shape == (2, 3)

    def test_sample_all_parameters_varies(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        g1 = torch.Generator().manual_seed(1)
        s1 = sample_all_parameters(layer, g1)
        g2 = torch.Generator().manual_seed(2)
        s2 = sample_all_parameters(layer, g2)
        # Different generators should give different sampled models
        assert not torch.allclose(s1.get_parameters()["W"].mean,
                                  s2.get_parameters()["W"].mean)

    def test_sample_all_parameters_gradient_flow(self):
        """Gradients should flow through sampled model back to original."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)

        x = torch.ones(2, 4)
        sampled = sample_all_parameters(layer)
        y = sampled(x)
        loss = y.sum()
        loss.backward()

        # Gradients should flow to original mean and log_sigma
        assert layer.W.mean.grad is not None
        assert torch.any(layer.W.mean.grad != 0)
        assert layer.W.log_sigma.grad is not None
        assert torch.any(layer.W.log_sigma.grad != 0)

    def test_gaussian_entropy_scalar(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        h = gaussian_entropy(layer)
        assert h.shape == ()
        assert torch.isfinite(h)

    def test_gaussian_entropy_deterministic_zero(self):
        """A fully deterministic model should have zero entropy."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False)
        h = gaussian_entropy(layer)
        assert torch.allclose(h, torch.tensor(0.0))

    def test_gaussian_entropy_value(self):
        """Check gaussian_entropy matches manual computation."""
        p = GaussianParameter(mean=torch.zeros(3), log_sigma=torch.tensor([-2.0, -0.5, 0.0]))

        class SingleParam(Module):
            def __init__(self):
                super().__init__()
                self.w = p

        m = SingleParam()
        h = gaussian_entropy(m)
        expected = torch.sum(torch.log(p.stdv))
        assert torch.allclose(h, expected)

    def test_laplacian_entropy_scalar(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, param_type=LaplacianParameter)
        h = laplacian_entropy(layer)
        assert h.shape == ()
        assert torch.isfinite(h)

    def test_laplacian_entropy_deterministic_zero(self):
        """A fully deterministic model should have zero Laplacian entropy."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False)
        h = laplacian_entropy(layer)
        assert torch.allclose(h, torch.tensor(0.0))

    def test_laplacian_entropy_value(self):
        """Check laplacian_entropy matches manual computation."""
        p = LaplacianParameter(mean=torch.zeros(4), log_scale=torch.tensor([-2.0, -1.0, -0.3, 0.0]))

        class SingleParam(Module):
            def __init__(self):
                super().__init__()
                self.w = p

        m = SingleParam()
        h = laplacian_entropy(m)
        expected = torch.sum(torch.log(p.scale))
        assert torch.allclose(h, expected)

    def test_laplacian_entropy_ignores_gaussian(self):
        """laplacian_entropy should only sum over Laplacian parameters."""
        torch.manual_seed(0)
        model = MixedModel()
        h = laplacian_entropy(model)
        params = model.get_parameters()
        expected = torch.tensor(0.0)
        for name, p in params.items():
            if isinstance(p, LaplacianParameter):
                expected = expected + torch.sum(torch.log(p.scale))
        assert torch.allclose(h, expected)

    def test_gaussian_entropy_ignores_laplacian(self):
        """gaussian_entropy should only sum over Gaussian parameters."""
        torch.manual_seed(0)
        model = MixedModel()
        h = gaussian_entropy(model)
        params = model.get_parameters()
        expected = torch.tensor(0.0)
        for name, p in params.items():
            if isinstance(p, GaussianParameter):
                expected = expected + torch.sum(torch.log(p.stdv))
        assert torch.allclose(h, expected)

    def test_get_parameter_count(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        counts = get_parameter_count(layer)
        assert counts["total"] == 15  # 12 + 3
        assert counts["bayesian"] == 15
        assert counts["deterministic"] == 0

    def test_get_parameter_groups(self):
        torch.manual_seed(0)
        model = MixedModel()
        groups = get_parameter_groups(model)
        assert len(groups["gaussian"]) > 0
        assert len(groups["laplacian"]) > 0
        assert len(groups["deterministic"]) > 0

    def test_flatten_means_module_level(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        m1 = flatten_means(layer)
        m2 = layer.flatten_means()
        assert torch.allclose(m1, m2)

    def test_flatten_stdvs_module_level(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        s1 = flatten_stdvs(layer)
        s2 = layer.flatten_stdvs()
        assert torch.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Mixed parameter model tests
# ---------------------------------------------------------------------------


class TestMixedModel:
    def test_forward_sample(self):
        torch.manual_seed(0)
        model = MixedModel()
        x = torch.ones(2, 4)
        y = model(x, sample=True)
        assert y.shape == (2, 1)

    def test_forward_deterministic(self):
        torch.manual_seed(0)
        model = MixedModel()
        x = torch.ones(2, 4)
        y = model(x, sample=False)
        assert y.shape == (2, 1)

    def test_parameter_types_correct(self):
        torch.manual_seed(0)
        model = MixedModel()
        params = model.get_parameters()
        assert isinstance(params["gauss_layer.W"], GaussianParameter)
        assert isinstance(params["laplace_layer.W"], LaplacianParameter)
        assert isinstance(params["det_layer.W"], DeterministicParameter)

    def test_sample_all_on_mixed(self):
        torch.manual_seed(0)
        model = MixedModel()
        sampled = sample_all_parameters(model)

        # Parameter types are preserved
        params = sampled.get_parameters()
        assert isinstance(params["gauss_layer.W"], GaussianParameter)
        assert isinstance(params["laplace_layer.W"], LaplacianParameter)
        assert isinstance(params["det_layer.W"], DeterministicParameter)

        # Bayesian means should differ from original
        assert not torch.allclose(
            params["gauss_layer.W"].mean, model.gauss_layer.W.mean
        )

        x = torch.ones(2, 4)
        y = sampled(x)
        assert y.shape == (2, 1)


# ---------------------------------------------------------------------------
# make_parameter tests
# ---------------------------------------------------------------------------


class TestMakeParameter:
    def test_gaussian_default(self):
        p = make_parameter(torch.zeros(3, 4))
        assert isinstance(p, GaussianParameter)
        assert p.shape == (3, 4)

    def test_scalar(self):
        p = make_parameter(torch.tensor(1.0))
        assert isinstance(p, GaussianParameter)
        assert p.shape == ()

    def test_deterministic(self):
        p = make_parameter(torch.ones(5), bayesian=False)
        assert isinstance(p, DeterministicParameter)
        assert torch.allclose(p.mean, torch.tensor(1.0))

    def test_laplacian(self):
        p = make_parameter(torch.eye(3), param_type=LaplacianParameter)
        assert isinstance(p, LaplacianParameter)
        assert p.shape == (3, 3)

    def test_init_log_sigma(self):
        p = make_parameter(torch.zeros(2), init_log_sigma=-3.0)
        assert torch.allclose(p.log_sigma, torch.tensor(-3.0))

    def test_in_custom_model(self):
        """make_parameter works inside a user-defined Module."""

        class MyModel(Module):
            def __init__(self):
                super().__init__()
                self.A = make_parameter(torch.zeros(3, 4))
                self.scale = make_parameter(torch.tensor(1.0), bayesian=False)

            def forward(self, x, *, sample=False, generator=None):
                if sample:
                    A = self.A.sample(generator)
                else:
                    A = self.A.mean
                return x @ A.T * self.scale.mean

        m = MyModel()
        params = m.get_parameters()
        assert "A" in params
        assert "scale" in params
        assert isinstance(params["A"], GaussianParameter)
        assert isinstance(params["scale"], DeterministicParameter)

        x = torch.ones(2, 4)
        y = m(x, sample=True)
        assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# sample_all_parameters immutability test
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_sample_all_does_not_modify_original(self):
        """sample_all_parameters must never modify the original model."""
        torch.manual_seed(0)
        model = BayesianLinear(4, 3)
        original_W_mean = model.W.mean.clone()
        original_b_mean = model.b.mean.clone()

        for i in range(50):
            g = torch.Generator().manual_seed(i)
            sampled = sample_all_parameters(model, g)
            assert not torch.allclose(sampled.get_parameters()["W"].mean, model.W.mean)

        assert torch.equal(model.W.mean, original_W_mean)
        assert torch.equal(model.b.mean, original_b_mean)


# ---------------------------------------------------------------------------
# Compile compatibility test (analogous to JIT tests)
# ---------------------------------------------------------------------------


class TestCompile:
    def test_forward_compiled(self):
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(2, 4)

        # torch.compile may not be available or may be a no-op on some platforms
        try:
            compiled_forward = torch.compile(layer)
            y = compiled_forward(x, sample=False)
            assert y.shape == (2, 3)
        except Exception:
            # torch.compile is optional; skip if not supported
            pytest.skip("torch.compile not available")

    def test_backward(self):
        """Gradients should work via standard backprop."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)

        x = torch.ones(1, 4)
        y = layer(x, sample=True)
        loss = y.sum()
        loss.backward()

        has_nonzero = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in layer.parameters()
            if p.requires_grad
        )
        assert has_nonzero

    def test_multiple_samples(self):
        """Multiple forward passes with sample=True should produce different outputs."""
        torch.manual_seed(0)
        layer = BayesianLinear(4, 3)
        x = torch.ones(4)

        ys = torch.stack([layer(x, sample=True) for _ in range(8)])
        assert ys.shape == (8, 3)
        assert not torch.allclose(ys[0], ys[1])
