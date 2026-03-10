import math
from typing import Callable


def stable_then_decay_lr(
    it: int,
    num_iterations: int,
    cooldown_frac: float,
) -> float:
    t = 1 - it / num_iterations # time remaining in training
    assert 1 >= t > 0
    # 1) constant lr for first part of training
    if t >= cooldown_frac:
        return 1.0
    # 2) then linear cooldown
    else:
        return t / cooldown_frac

def constant_lr(
    it: int,
) -> float:
    return 1.0

def warmup_then_stable_then_decay_lr(
    it: int,
    num_iterations: int,
    cooldown_frac: float,
    warmup_frac: float,
) -> float:
    t = it / num_iterations
    if t < warmup_frac:
        return t / warmup_frac
    elif t < (1 - cooldown_frac):
        return 1.0
    else:
        lr = (1 - t) / cooldown_frac
        if lr < 0.0:
            return 0.0
        return lr

def cosine_with_warmup_lr(
    it: int,
    num_iterations: int,
    warmup_frac: float,
    min_lr_frac: float = 0.0,
) -> float:
    t = it / num_iterations
    if t < warmup_frac:
        return t / warmup_frac
    elif t > 1.0:
        return min_lr_frac
    decay_t = (t - warmup_frac) / (1.0 - warmup_frac)
    return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + math.cos(math.pi * decay_t))


LR_SCHEDULER_FUNCTION_MAPPING: dict[str, Callable] = {
    "stable_then_decay": stable_then_decay_lr,
    "constant": constant_lr,
    "wsd": warmup_then_stable_then_decay_lr,
    "cosine_with_warmup": cosine_with_warmup_lr,
}