import math

import torch
from torch import nn

from sparse_neurons.conversion import convert_linear_layers, iter_group_ard_layers
from sparse_neurons.layers import GroupARDLinear, TwoSidedGroupARDLinear
from sparse_neurons.models import DeterministicLeNet300100
from sparse_neurons.experiments.train_mnist import make_model
from sparse_neurons.experiments.train_mnist import kl_weight
from argparse import Namespace


def test_mean_initialization_has_linear_bounds() -> None:
    torch.manual_seed(7)
    layer = GroupARDLinear(20, 5)
    bound = 1 / math.sqrt(20)
    assert torch.all(layer.weight_mu.abs() <= bound)
    assert torch.all(layer.bias_mu.abs() <= bound)


def test_exact_ml_log_lambda_update() -> None:
    layer = GroupARDLinear(3, 2)
    with torch.no_grad():
        layer.weight_mu.fill_(1.0)
        layer.bias_mu.fill_(1.0)
        layer.weight_log_variance.fill_(math.log(0.5))
        layer.bias_log_variance.fill_(math.log(0.5))
    layer.update_log_lambda()
    expected_energy = 4 * 1.5
    assert torch.allclose(layer.log_lambda, torch.full((2,), math.log(4 / expected_energy)))


def test_deterministic_forward_matches_linear() -> None:
    layer = GroupARDLinear(3, 2)
    x = torch.randn(4, 3)
    expected = nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.equal(layer(x, sample=False), expected)


def test_kl_is_nonnegative_at_exact_update() -> None:
    layer = GroupARDLinear(8, 4)
    layer.update_log_lambda()
    assert layer.kl_divergence().item() >= 0.0


def test_from_linear_preserves_posterior_mean_output() -> None:
    original = nn.Linear(6, 3, dtype=torch.float64)
    converted = GroupARDLinear.from_linear(original)
    x = torch.randn(4, 6, dtype=torch.float64)
    assert torch.equal(original(x), converted(x, sample=False))
    assert converted.weight_mu.dtype == original.weight.dtype


def test_convert_nested_network_copies_and_preserves_mean_output() -> None:
    original = nn.Sequential(
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Sequential(nn.Linear(4, 2)),
    ).eval()
    converted = convert_linear_layers(original).eval()
    x = torch.randn(5, 6)
    assert torch.equal(original(x), converted(x))
    assert len(list(iter_group_ard_layers(converted))) == 2
    assert isinstance(original[0], nn.Linear)
    assert not isinstance(original[0], GroupARDLinear)


def test_conversion_preserves_shared_linear_modules() -> None:
    class SharedNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(3, 3)
            self.first = shared
            self.second = shared

    converted = convert_linear_layers(SharedNetwork())
    assert isinstance(converted.first, GroupARDLinear)
    assert converted.first is converted.second


def test_convert_complete_lenet_preserves_mean_output() -> None:
    original = DeterministicLeNet300100().eval()
    converted = convert_linear_layers(original).eval()
    images = torch.randn(3, 1, 28, 28)
    assert torch.equal(original(images), converted(images))


def test_two_sided_conversion_preserves_mean_output() -> None:
    original = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    converted = convert_linear_layers(original, ard_type="two_sided").eval()
    x = torch.randn(3, 5)
    assert torch.equal(original(x), converted(x))
    assert all(isinstance(layer, TwoSidedGroupARDLinear) for layer in iter_group_ard_layers(converted))


def test_two_sided_ml_updates_are_stationary() -> None:
    layer = TwoSidedGroupARDLinear(7, 5, m_step_sweeps=20)
    layer.update_log_lambda()
    old_in = layer.log_lambda_in.clone()
    old_out = layer.log_lambda_out.clone()
    layer.update_log_lambda()
    assert torch.allclose(layer.log_lambda_in, old_in, atol=2e-4)
    assert torch.allclose(layer.log_lambda_out, old_out, atol=2e-4)


def test_two_sided_fixed_output_scale_remains_one() -> None:
    layer = TwoSidedGroupARDLinear(7, 5, fix_output_scale=True)
    with torch.no_grad():
        layer.log_lambda_out.fill_(4.0)
        layer.weight_mu.normal_()
    layer.update_log_lambda()
    assert torch.equal(layer.log_lambda_out, torch.zeros_like(layer.log_lambda_out))


def test_relative_variance_initialization_tracks_copied_means() -> None:
    original = nn.Linear(4, 3)
    converted = TwoSidedGroupARDLinear.from_linear(
        original,
        initial_relative_variance=1e-6,
        variance_floor=1e-12,
    )
    expected = 1e-12 + 1e-6 * original.weight.square()
    assert torch.allclose(converted.weight_log_variance.exp(), expected, rtol=1e-5)


def test_unit_ratio_mixture_matches_single_gaussian_kl() -> None:
    plain = TwoSidedGroupARDLinear(5, 3)
    mixture = TwoSidedGroupARDLinear(5, 3, mixture_spike_ratio=1.0)
    mixture.load_state_dict(plain.state_dict(), strict=False)
    plain.update_log_lambda()
    mixture.update_log_lambda()
    assert torch.allclose(mixture.kl_divergence(), plain.kl_divergence(), atol=1e-4)


def test_mixture_responsibilities_and_kl_are_valid() -> None:
    layer = TwoSidedGroupARDLinear(7, 4, mixture_spike_ratio=1e-2)
    layer.update_log_lambda()
    assert torch.all((layer.spike_responsibility >= 0) & (layer.spike_responsibility <= 1))
    assert 0 < layer.spike_probability.item() < 1
    assert torch.isfinite(layer.kl_divergence())
    assert layer.kl_divergence().item() >= 0


def test_pretrained_conversion_can_keep_final_layer_dense(tmp_path) -> None:
    original = DeterministicLeNet300100()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": original.state_dict(), "epoch": 1}, checkpoint)
    args = Namespace(
        pretrained_checkpoint=checkpoint,
        initial_log_variance=-12.0,
        initial_relative_variance=None,
        ard_type="two_sided",
        mixture_spike_ratio=1e-2,
        mixture_spike_variance=None,
        dense_final_layer=True,
        hidden_sizes=(300, 100),
    )
    converted = make_model(args, torch.device("cpu")).eval()
    x = torch.randn(3, 1, 28, 28)
    assert isinstance(converted.layers[-1], nn.Linear)
    assert not isinstance(converted.layers[-1], TwoSidedGroupARDLinear)
    assert torch.equal(original.eval()(x), converted(x))


def test_configurable_deterministic_hidden_sizes() -> None:
    model = DeterministicLeNet300100((300, 300))
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert model.layers[1].out_features == 300
    assert model.layers[3].out_features == 300


def test_shared_spike_variance_has_exact_m_step() -> None:
    layer = TwoSidedGroupARDLinear(
        7, 4, mixture_spike_variance=1e-4, m_step_sweeps=1
    )
    layer.update_log_lambda()
    responsibility = layer.spike_responsibility
    expected = (responsibility * layer.weight_second_moment()).sum() / responsibility.sum()
    assert torch.allclose(layer.log_spike_variance.exp(), expected, rtol=1e-5)
    assert torch.isfinite(layer.kl_divergence())
    assert layer.kl_divergence().item() >= 0


def test_mixture_parameterizations_are_mutually_exclusive() -> None:
    try:
        TwoSidedGroupARDLinear(
            3, 2, mixture_spike_ratio=1e-2, mixture_spike_variance=1e-4
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected mutually exclusive mixture parameters")


def test_long_run_kl_schedule_reaches_full_strength() -> None:
    assert kl_weight(30, zero_epochs=30, warmup_epochs=200) == 0.0
    assert kl_weight(130, zero_epochs=30, warmup_epochs=200) == 0.5
    assert kl_weight(230, zero_epochs=30, warmup_epochs=200) == 1.0
    assert kl_weight(300, zero_epochs=30, warmup_epochs=200) == 1.0
