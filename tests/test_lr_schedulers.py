"""Tests for LR scheduler functions in core.optimizers.lr_scheduler_utils."""

import math

import pytest

from core.optimizers.lr_scheduler_utils import (
    LR_SCHEDULER_FUNCTION_MAPPING,
    constant_lr,
    cosine_with_warmup_lr,
    stable_then_decay_lr,
    warmup_then_stable_then_decay_lr,
)


class TestConstantLR:
    def test_always_returns_one(self):
        for it in range(100):
            assert constant_lr(it) == 1.0


class TestStableThenDecay:
    def test_stable_phase(self):
        n = 100
        for it in range(1, 70):
            assert stable_then_decay_lr(it, n, cooldown_frac=0.3) == 1.0

    def test_decay_phase_decreases(self):
        n = 100
        prev = 1.0
        for it in range(71, 100):
            lr = stable_then_decay_lr(it, n, cooldown_frac=0.3)
            assert lr < prev
            prev = lr

    def test_approaches_zero_at_end(self):
        n = 1000
        lr = stable_then_decay_lr(n - 1, n, cooldown_frac=0.1)
        assert lr < 0.02


class TestWarmupStableDecay:
    def test_starts_near_zero(self):
        lr = warmup_then_stable_then_decay_lr(1, 1000, cooldown_frac=0.1, warmup_frac=0.1)
        assert lr < 0.02

    def test_warmup_ramps_up(self):
        n = 1000
        prev = 0.0
        for it in range(1, 100):
            lr = warmup_then_stable_then_decay_lr(it, n, cooldown_frac=0.1, warmup_frac=0.1)
            assert lr >= prev
            prev = lr

    def test_stable_phase(self):
        n = 1000
        for it in range(100, 900):
            lr = warmup_then_stable_then_decay_lr(it, n, cooldown_frac=0.1, warmup_frac=0.1)
            assert lr == 1.0

    def test_decay_phase(self):
        n = 1000
        prev = 1.0
        for it in range(901, 1000):
            lr = warmup_then_stable_then_decay_lr(it, n, cooldown_frac=0.1, warmup_frac=0.1)
            assert lr <= prev
            prev = lr

    def test_never_negative(self):
        n = 100
        for it in range(1, n + 1):
            lr = warmup_then_stable_then_decay_lr(it, n, cooldown_frac=0.1, warmup_frac=0.1)
            assert lr >= 0.0


class TestCosineWithWarmup:
    def test_starts_near_zero(self):
        lr = cosine_with_warmup_lr(1, 1000, warmup_frac=0.1)
        assert lr < 0.02

    def test_reaches_one_at_warmup_end(self):
        lr = cosine_with_warmup_lr(100, 1000, warmup_frac=0.1)
        assert abs(lr - 1.0) < 1e-6

    def test_warmup_ramps_linearly(self):
        n = 1000
        prev = 0.0
        for it in range(1, 100):
            lr = cosine_with_warmup_lr(it, n, warmup_frac=0.1)
            assert lr >= prev
            prev = lr

    def test_decays_after_warmup(self):
        n = 1000
        prev = 1.0
        for it in range(101, n):
            lr = cosine_with_warmup_lr(it, n, warmup_frac=0.1)
            assert lr <= prev + 1e-9
            prev = lr

    def test_reaches_zero_at_end(self):
        lr = cosine_with_warmup_lr(1000, 1000, warmup_frac=0.1)
        assert abs(lr) < 1e-6

    def test_min_lr_frac(self):
        lr = cosine_with_warmup_lr(1000, 1000, warmup_frac=0.1, min_lr_frac=0.1)
        assert abs(lr - 0.1) < 1e-6

    def test_midpoint_is_half(self):
        """At the midpoint of the cosine phase, LR should be ~0.5."""
        n = 1000
        warmup_frac = 0.1
        mid = int(100 + (1000 - 100) / 2)
        lr = cosine_with_warmup_lr(mid, n, warmup_frac=warmup_frac)
        assert abs(lr - 0.5) < 0.01

    def test_symmetric_around_midpoint(self):
        n = 1000
        warmup_frac = 0.1
        warmup_end = 100
        quarter = warmup_end + int((n - warmup_end) * 0.25)
        three_quarter = warmup_end + int((n - warmup_end) * 0.75)
        lr_q = cosine_with_warmup_lr(quarter, n, warmup_frac=warmup_frac)
        lr_3q = cosine_with_warmup_lr(three_quarter, n, warmup_frac=warmup_frac)
        assert abs(lr_q + lr_3q - 1.0) < 0.02


class TestSchedulerMapping:
    def test_cosine_with_warmup_registered(self):
        assert "cosine_with_warmup" in LR_SCHEDULER_FUNCTION_MAPPING

    def test_all_expected_schedulers_present(self):
        expected = {"stable_then_decay", "constant", "wsd", "cosine_with_warmup"}
        assert set(LR_SCHEDULER_FUNCTION_MAPPING.keys()) == expected
