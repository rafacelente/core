def __getattr__(name: str):
    if name == "CoreLightningModel":
        from .lightning_model import CoreLightningModel
        return CoreLightningModel
    if name == "LogCallback":
        from .callbacks.log_callback import LogCallback
        return LogCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CoreLightningModel", "LogCallback"]
