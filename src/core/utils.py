from collections.abc import MutableMapping

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