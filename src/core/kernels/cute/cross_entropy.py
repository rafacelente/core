from typing import Tuple

from quack.cross_entropy import cross_entropy_fwd, cross_entropy_bwd
import torch

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
        loss, lse = cross_entropy_fwd(
            logits, labels, ignore_index=ignore_index, return_lse=True
        )
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        logits, labels, lse = ctx.saved_tensors
        # Ensure grad_loss is contiguous - the kernel requires stride=1
        grad_loss = grad_loss.contiguous()
        grad_logits = cross_entropy_bwd(
            logits, labels, grad_loss, lse, ignore_index=ctx.ignore_index
        )
        return grad_logits, None, None