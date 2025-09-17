import math
from typing import Optional
import os

import torch


# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.0,
        muon_params: Optional[list[torch.nn.Parameter]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[list[torch.nn.Parameter]] = None,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        rms_update_ratio: float = 0.2,  # from https://arxiv.org/pdf/2502.16982. Should range from 0.2 to 0.4
        skip_adjust_lr: bool = False,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            rms_update_ratio=rms_update_ratio,
            skip_adjust_lr=skip_adjust_lr,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        if muon_params is not None:
            for p in muon_params:
                assert p.ndim == 2, p.ndim
                self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape, rms_update_ratio):
        A, B = param_shape[:2]
        adjusted_ratio = rms_update_ratio * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = (
                    self.adjust_lr_for_muon(lr, p.shape, group["rms_update_ratio"])
                    if not group["skip_adjust_lr"]
                    else lr
                )

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def configure_muon(
    model: torch.nn.Module, lr: float, betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.01, print_param_names: bool = True
) -> list[torch.optim.Optimizer]:
    """
    Helper function to configure Muon optimizer for language/transactional model training with
    AdamW optimizer for non-matrix parameters.

    Args:
        model: The model to configure the optimizer for.
        lr: The learning rate for the optimizer.
        betas: The betas for the AdamW optimizer.
        weight_decay: The weight decay for the AdamW optimizer.
    """
    embed_params: list[torch.nn.Parameter] = []
    lm_head_params: list[torch.nn.Parameter] = []
    matrix_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []

    assigned_param_ids: set[int] = set()

    scalar_param_names: list[str] = []
    muon_param_names: list[str] = []

    for possible_attr in ("embed", "embeddings", "wte"):
        if hasattr(model, possible_attr):
            params = list(getattr(model, possible_attr).parameters())
            embed_params.extend(params)
            scalar_param_names.extend([f"{possible_attr}.{i}" for i in range(len(params))])
            assigned_param_ids.update(id(p) for p in params)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        lm_head_params.append(model.lm_head.weight)
        scalar_param_names.append("lm_head.weight")
        assigned_param_ids.add(id(model.lm_head.weight))

    for name, p in model.named_parameters():
        if id(p) in assigned_param_ids:
            continue

        if p.ndim == 2:
            matrix_params.append(p)
            muon_param_names.append(name)
            assigned_param_ids.add(id(p))
        else:
            scalar_params.append(p)
            scalar_param_names.append(name)
            assigned_param_ids.add(id(p))

    assert len(assigned_param_ids) == len(list(model.parameters())), "Some parameters were not assigned to any optimizer or were duplicated."

    adamw_params = scalar_params + lm_head_params + embed_params

    if print_param_names:
        print("Muon params:")
        for name in muon_param_names:
            print(f"  {name}")
        print("AdamW params:")
        for name in scalar_param_names:
            print(f"  {name}")

    return Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=matrix_params,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=adamw_params,
        adamw_betas=betas,
        adamw_eps=1e-8,
        rms_update_ratio=0.2,
        skip_adjust_lr=False,
    )