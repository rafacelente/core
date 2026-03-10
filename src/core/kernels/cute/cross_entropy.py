from typing import Tuple

import torch


def _import_quack_cross_entropy():
    try:
        from quack.cross_entropy import cross_entropy_fwd, cross_entropy_bwd
        return cross_entropy_fwd, cross_entropy_bwd
    except ImportError as exc:
        raise ImportError(
            "Fused cross-entropy requires the 'quack-kernels' package. "
            "Install it with: pip install 'core[gpu]'"
        ) from exc


class _FusedCrossEntropyFunction(torch.autograd.Function):
    """Custom autograd function that ensures gradient tensors are contiguous.

    The quack cross entropy backward kernel requires contiguous tensors, but
    PyTorch's autograd may create gradient tensors with stride=0 (broadcasted).
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        cross_entropy_fwd, _ = _import_quack_cross_entropy()
        loss, lse = cross_entropy_fwd(
            logits, labels, ignore_index=ignore_index, return_lse=True
        )
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        _, cross_entropy_bwd = _import_quack_cross_entropy()
        logits, labels, lse = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_logits = cross_entropy_bwd(
            logits, labels, grad_loss, lse, ignore_index=ctx.ignore_index
        )
        return grad_logits, None, None