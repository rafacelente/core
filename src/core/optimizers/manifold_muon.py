import math
import torch

ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# safety factor for numerical stability (but exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign function:
    https://arxiv.org/abs/2505.16932
    """
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x /= x.norm(dim=(-2, -1), keepdim=True) * 1.01
    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        # goal is to compute x = a x + b S x + c S^2 x
        # we can break this up into: x = (a I + (b I + c S) S) x
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    x = torch.nan_to_num(x)
    return x.float()

@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W


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


class ManifoldMuon(torch.optim.Optimizer):
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
        skip_adjust_lr: bool = True,
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
            #       Manifold Muon     #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                p.data = manifold_muon(p.data, g, eta=lr, alpha=0.01, steps=100, tol=1e-6)


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


def configure_manifold_muon(
    model: torch.nn.Module,
    lr: float,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    rms_update_ratio: float = 0.2,
    skip_adjust_lr: bool = False,
    ns_steps: int = 5,
    print_param_names: bool = False
) -> list[torch.optim.Optimizer]:
    """
    Helper function to configure Muon optimizer for language/transactional model training with
    AdamW optimizer for non-matrix parameters.

    Args:
        model: The model to configure the optimizer for.
        lr: The learning rate for the optimizer.
        adamw_betas: The betas for the AdamW optimizer.
        muon_momentum: The momentum for the Muon optimizer.
        muon_nesterov: Whether to use Nesterov-style momentum in the Muon optimizer.
        weight_decay: The weight decay for the AdamW optimizer.
        skip_adjust_lr: Whether to skip adjusting the learning rate for the Muon optimizer.
        ns_steps: The number of Newton-Schulz iterations to run for the Muon optimizer.
        print_param_names: Whether to print the names of the parameters in the Muon and AdamW optimizers.
        rms_update_ratio: The RMS update ratio for the Muon optimizer.
    """
    embed_params: list[torch.nn.Parameter] = []
    lm_head_params: list[torch.nn.Parameter] = []
    matrix_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []

    assigned_param_ids: set[int] = set()

    scalar_param_names: list[str] = []
    muon_param_names: list[str] = []

    for possible_attr in ("embed", "embeddings", "wte", "embed_tokens"):
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
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=ns_steps,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=1e-8,
        rms_update_ratio=rms_update_ratio,
        skip_adjust_lr=skip_adjust_lr,
    )