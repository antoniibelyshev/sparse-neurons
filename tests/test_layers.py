import math
from argparse import Namespace

import torch
from torch import nn

from sparse_neurons.conversion import (
    convert_conv2d_layers,
    convert_linear_layers,
    iter_group_ard_layers,
)
from sparse_neurons.ema import ParameterEMA
from sparse_neurons.experiments.evaluate_neuron_importance import neuron_importance
from sparse_neurons.experiments.train_mnist import kl_weight, make_model
from sparse_neurons.layers import TwoSidedGroupARDConv2d, TwoSidedGroupARDLinear
from sparse_neurons.models import DeterministicLeNet300100


def test_mean_initialization_has_linear_bounds() -> None:
    torch.manual_seed(7)
    layer = TwoSidedGroupARDLinear(20, 5)
    bound = 1 / math.sqrt(20)
    assert torch.all(layer.weight_mu.abs() <= bound)
    assert torch.all(layer.bias_mu.abs() <= bound)


def test_deterministic_forward_matches_linear() -> None:
    layer = TwoSidedGroupARDLinear(3, 2)
    x = torch.randn(4, 3)
    expected = nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.equal(layer(x, sample=False), expected)


def test_conversion_copies_network_and_preserves_mean_output() -> None:
    original = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    converted = convert_linear_layers(original).eval()
    x = torch.randn(5, 6)
    assert torch.equal(original(x), converted(x))
    assert len(list(iter_group_ard_layers(converted))) == 2
    assert isinstance(original[0], nn.Linear)


def test_conversion_preserves_shared_linear_modules() -> None:
    class SharedNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(3, 3)
            self.first = shared
            self.second = shared

    converted = convert_linear_layers(SharedNetwork())
    assert isinstance(converted.first, TwoSidedGroupARDLinear)
    assert converted.first is converted.second


def test_augmented_ml_updates_are_stationary() -> None:
    layer = TwoSidedGroupARDLinear(7, 5, m_step_sweeps=20)
    layer.update_log_lambda()
    old_in = layer.log_lambda_in.clone()
    old_out = layer.log_lambda_out.clone()
    layer.update_log_lambda()
    assert torch.allclose(layer.log_lambda_in, old_in, atol=2e-4)
    assert torch.allclose(layer.log_lambda_out, old_out, atol=2e-4)


def test_relative_std_initialization_tracks_copied_means() -> None:
    original = nn.Linear(4, 3)
    converted = TwoSidedGroupARDLinear.from_linear(
        original, initial_relative_std=1e-2, variance_floor=1e-12
    )
    expected = (1e-2 * original.weight.abs()).clamp_min(1e-6)
    assert torch.allclose(converted.weight_log_std.exp(), expected, rtol=1e-5)


def test_mixture_responsibilities_and_kl_are_valid() -> None:
    layer = TwoSidedGroupARDLinear(7, 4, mixture_spike_variance=1e-4)
    layer.update_log_lambda()
    assert torch.all((layer.spike_responsibility >= 0) & (layer.spike_responsibility <= 1))
    assert layer.spike_responsibility.shape == (4, 8)
    assert 0 < layer.spike_probability.item() < 1
    assert torch.isfinite(layer.kl_divergence())
    assert layer.kl_divergence().item() >= 0
    assert torch.allclose(
        layer.elementwise_kl_divergence().sum(), layer.kl_divergence()
    )


def test_pretrained_conversion_keeps_final_layer_dense(tmp_path) -> None:
    original = DeterministicLeNet300100()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": original.state_dict(), "epoch": 1}, checkpoint)
    args = Namespace(
        pretrained_checkpoint=checkpoint,
        initial_relative_std=1e-2,
        mixture_spike_variance=1e-4,
        hidden_sizes=(300, 100),
    )
    converted = make_model(args, torch.device("cpu")).eval()
    x = torch.randn(3, 1, 28, 28)
    assert isinstance(converted.layers[-1], nn.Linear)
    assert torch.equal(original.eval()(x), converted(x))


def test_shared_spike_variance_has_exact_m_step() -> None:
    layer = TwoSidedGroupARDLinear(7, 4, mixture_spike_variance=1e-4, m_step_sweeps=1)
    layer.update_log_lambda()
    responsibility = layer.spike_responsibility
    expected = (
        responsibility * layer.augmented_weight_second_moment()
    ).sum() / responsibility.sum()
    assert torch.allclose(layer.log_spike_variance.exp(), expected, rtol=1e-5)


def test_long_run_kl_schedule_reaches_full_strength() -> None:
    assert kl_weight(1, zero_epochs=0, warmup_epochs=200) == 0.005
    assert kl_weight(100, zero_epochs=0, warmup_epochs=200) == 0.5
    assert kl_weight(200, zero_epochs=0, warmup_epochs=200) == 1.0
    assert kl_weight(300, zero_epochs=0, warmup_epochs=200) == 1.0


def test_neuron_importance_is_maximum_augmented_weight_snr() -> None:
    layer = TwoSidedGroupARDLinear(3, 2)
    with torch.no_grad():
        layer.weight_mu.copy_(torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]))
        layer.weight_log_std.copy_(
            0.5 * torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 0.25]]).log()
        )
        layer.bias_mu.copy_(torch.tensor([2.0, 0.0]))
        layer.bias_log_std.zero_()
    assert torch.allclose(neuron_importance(layer), torch.tensor([4.0, 2.0]))


def test_bias_has_learned_augmented_input_scale() -> None:
    layer = TwoSidedGroupARDLinear(3, 2)
    assert layer.log_lambda_in.shape == (4,)
    with torch.no_grad():
        layer.bias_mu.fill_(3.0)
        layer.bias_log_std.fill_(-4.0)
    layer.update_log_lambda()
    assert layer.log_lambda_in[-1].item() != 0.0


def test_parameter_ema_updates_and_copies_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ParameterEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(7.0)
    ema.copy_to(model)
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_conv_conversion_preserves_mean_output() -> None:
    original = nn.Conv2d(3, 5, 3, padding=1).eval()
    converted = convert_conv2d_layers(original).eval()
    x = torch.randn(2, 3, 8, 8)
    assert isinstance(converted, TwoSidedGroupARDConv2d)
    assert torch.equal(original(x), converted(x))


def test_conv_augmented_shapes_and_kl() -> None:
    layer = TwoSidedGroupARDConv2d(3, 5, 3)
    assert layer.spike_responsibility.shape == (5, 28)
    assert layer.log_lambda_in.shape == (4,)
    layer.update_log_lambda()
    assert torch.isfinite(layer.kl_divergence())
    assert torch.allclose(
        layer.elementwise_kl_divergence().sum(), layer.kl_divergence()
    )
