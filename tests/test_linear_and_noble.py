import pytest
import torch

from core.modules.activation import Activation, ActivationType, CosNet
from core.modules.noble import Noble, LinearNoble, NobleConfig
from core.modules.linear import LinearConfig, LinearType


class TestCosNet:
    def test_forward_shape(self):
        r = 32
        cosnet = CosNet(r=r)
        cosnet.reset_parameters()
        x = torch.randn(2, 10, r)
        out = cosnet(x)
        assert out.shape == x.shape

    def test_activation_build_cosnet(self):
        act = Activation.build(ActivationType.COSNET, r=16)
        assert isinstance(act, CosNet)

    def test_omega_range(self):
        cosnet = CosNet(r=8, omega_min=0.5, omega_max=1.5)
        cosnet.reset_parameters()
        assert cosnet.omega_1.min() >= 0.5
        assert cosnet.omega_1.max() <= 1.5


class TestNoble:
    def test_forward_shape(self):
        r, d_in, d_out = 16, 64, 128
        act = Activation.build(ActivationType.SILU)
        noble = Noble(r=r, d_in=d_in, d_out=d_out, activation=act)
        x = torch.randn(2, 10, d_in)
        out = noble(x)
        assert out.shape == (2, 10, d_out)

    def test_noble_with_cosnet(self):
        r, d_in, d_out = 16, 64, 128
        act = Activation.build(ActivationType.COSNET, r=r)
        noble = Noble(r=r, d_in=d_in, d_out=d_out, activation=act)
        x = torch.randn(2, 10, d_in)
        out = noble(x)
        assert out.shape == (2, 10, d_out)


class TestLinearNoble:
    def test_forward_shape(self):
        r, d_in, d_out = 16, 64, 128
        act = Activation.build(ActivationType.SILU)
        ln = LinearNoble(r=r, d_in=d_in, d_out=d_out, activation=act, bias=False)
        x = torch.randn(2, 10, d_in)
        out = ln(x)
        assert out.shape == (2, 10, d_out)

    def test_weight_property(self):
        r, d_in, d_out = 16, 64, 128
        act = Activation.build(ActivationType.SILU)
        ln = LinearNoble(r=r, d_in=d_in, d_out=d_out, activation=act, bias=False)
        assert ln.weight is ln.linear.weight
        assert ln.weight.shape == (d_out, d_in)


class TestLinearConfig:
    def test_default_builds_nn_linear(self):
        cfg = LinearConfig()
        m = cfg.build(in_features=64, out_features=128, bias=False)
        assert isinstance(m, torch.nn.Linear)
        assert m.weight.shape == (128, 64)

    def test_noble_builds_linear_noble(self):
        cfg = LinearConfig(
            type=LinearType.NOBLE,
            noble=NobleConfig(r=16, activation_type=ActivationType.COSNET),
        )
        m = cfg.build(in_features=64, out_features=128, bias=False)
        assert isinstance(m, LinearNoble)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 128)

    def test_noble_config_apply_to(self):
        nc = NobleConfig(r=8, apply_to=["att", "ff"])
        assert "att" in nc.apply_to
        assert "ff" in nc.apply_to
        assert "lm_head" not in nc.apply_to
