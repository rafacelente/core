"""Tests for shared utility functions in core.utils."""

import torch
import pytest

from core.utils import justnorm, normalize_matrix


class TestJustnorm:
    def test_output_is_unit_norm(self):
        x = torch.randn(4, 8)
        out = justnorm(x, dim=-1)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_preserves_dtype(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        out = justnorm(x)
        assert out.dtype == torch.bfloat16

    def test_custom_dim(self):
        x = torch.randn(3, 5)
        out = justnorm(x, dim=0)
        norms = out.norm(p=2, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_1d_tensor(self):
        x = torch.randn(16)
        out = justnorm(x, dim=0)
        assert torch.allclose(out.norm(), torch.tensor(1.0), atol=1e-5)


class TestNormalizeMatrix:
    def test_in_place_unit_norm(self):
        m = torch.randn(4, 8)
        normalize_matrix(m)
        norms = m.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_with_scale(self):
        m = torch.randn(4, 8)
        scale = 3.0
        normalize_matrix(m, scale=scale)
        norms = m.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, scale), atol=1e-4)

    def test_dim_0(self):
        m = torch.randn(4, 8)
        normalize_matrix(m, dim=0)
        norms = m.norm(p=2, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_return_value(self):
        m = torch.randn(4, 8)
        result = normalize_matrix(m)
        assert result is None

    def test_identity_with_normalized_attention(self):
        """Simulates the NormalizedAttention.normalize_matrices pattern."""
        w_q = torch.randn(64, 32)
        w_o = torch.randn(32, 64)
        normalize_matrix(w_q)
        normalize_matrix(w_o, dim=0)
        assert torch.allclose(w_q.norm(p=2, dim=-1), torch.ones(64), atol=1e-5)
        assert torch.allclose(w_o.norm(p=2, dim=0), torch.ones(64), atol=1e-5)

    def test_scale_matches_normalized_mlp_pattern(self):
        """Simulates NormalizedMLP: normalize + scale by sqrt(d_model)."""
        import math
        d_model = 64
        w = torch.randn(128, d_model)
        normalize_matrix(w, scale=math.sqrt(d_model))
        row_norms = w.norm(p=2, dim=-1)
        expected = torch.full_like(row_norms, math.sqrt(d_model))
        assert torch.allclose(row_norms, expected, atol=1e-4)
