from collections.abc import MutableMapping
from enum import Enum
import torch


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    A cache for buffers that are used in the model.
    """


def get_default_device() -> torch.device:
    """
    Get the default device.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class DType(str, Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def to_torch_dtype(self) -> torch.dtype:
        if self == DType.FLOAT32 or self == DType.FP32:
            return torch.float32
        elif self == DType.BFLOAT16 or self == DType.BF16:
            return torch.bfloat16
        elif self == DType.FLOAT16 or self == DType.FP16:
            return torch.float16
        else:
            raise ValueError(f"Invalid dtype: {self}")


def is_sm100_or_higher() -> bool:
    """
    Check if the current device is a SM100 or higher.
    """
    return torch.cuda.get_device_capability() >= (10, 0)


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalize *x* along *dim*, keeping the original dtype."""
    return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)


@torch.no_grad()
def normalize_matrix(
    m: torch.Tensor, dim: int = -1, *, scale: float | None = None
) -> None:
    """In-place L2-normalize *m* along *dim*, optionally rescaling by *scale*."""
    normed = justnorm(m, dim=dim)
    if scale is not None:
        normed = normed * scale
    m.copy_(normed)