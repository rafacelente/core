from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_FUSED_KERNEL_OVERRIDES = {
    ("layer_norm", "layer_norm_type"): ("rms_fast", "rms"),
    ("output_norm", "layer_norm_type"): ("rms_fast", "rms"),
    ("loss", "type"): ("fused_cross_entropy", "cross_entropy"),
    ("attention", "use_flash_attn_4"): (True, False),
    ("attention.rope", "type"): ("fused", "default"),
}


def _defuse_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Replace fused-kernel settings with portable equivalents."""
    for path_key, (fused_val, portable_val) in _FUSED_KERNEL_OVERRIDES.items():
        section_path, field = path_key

        parts = section_path.split(".")
        d = cfg
        for p in parts:
            if not isinstance(d, dict) or p not in d:
                d = None
                break
            d = d[p]

        if d is not None and isinstance(d, dict) and d.get(field) == fused_val:
            logger.info("  Swapping %s.%s: %s -> %s", section_path, field, fused_val, portable_val)
            d[field] = portable_val

    cfg["label_weights_path"] = None
    return cfg


def _extract_core_config_from_yaml(path: str) -> Dict[str, Any]:
    """Return the ``core_config`` dict from a training YAML.

    Handles two layouts:
    1. Training YAML with ``model.core_config`` nested section.
    2. A bare ``CoreConfig`` YAML (has ``n_layers`` at top level).
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if "model" in raw and isinstance(raw["model"], dict):
        model_section = raw["model"]
        if "core_config" in model_section:
            return model_section["core_config"]

    if "n_layers" in raw and "d_model" in raw:
        return raw

    raise ValueError(
        f"Could not locate a CoreConfig in {path}. "
        "Expected either a training YAML with model.core_config or a "
        "bare CoreConfig YAML with n_layers/d_model at the top level."
    )


def _extract_core_config_from_checkpoint(ckpt: dict) -> Optional[Dict[str, Any]]:
    """Try to pull the CoreConfig dict from Lightning hyper_parameters."""
    hp = ckpt.get("hyper_parameters") or ckpt.get("hparams")
    if hp is None:
        return None

    config_obj = hp.get("config")
    if config_obj is None:
        return None

    if isinstance(config_obj, dict):
        return config_obj
    if hasattr(config_obj, "model_dump"):
        return config_obj.model_dump()
    if hasattr(config_obj, "dict"):
        return config_obj.dict()
    return None


def _extract_state_dict(ckpt: dict, is_lightning: bool) -> Dict[str, torch.Tensor]:
    """Return a state dict with keys relative to ``CoreModel``."""
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if not is_lightning:
        return sd

    prefix = "model."
    stripped: Dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = val
        else:
            stripped[key] = val
    return stripped


def convert(
    checkpoint_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    dtype: Optional[str] = None,
    keep_fused_kernels: bool = False,
) -> None:
    ckpt_path = Path(checkpoint_path)
    is_lightning = ckpt_path.suffix == ".ckpt"

    logger.info("Loading checkpoint from %s …", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ---- Resolve CoreConfig dict ----
    core_config_dict: Optional[Dict[str, Any]] = None

    if config_path is not None:
        logger.info("Loading config from %s", config_path)
        core_config_dict = _extract_core_config_from_yaml(config_path)
    else:
        logger.info("No --config provided; trying checkpoint hyper_parameters …")
        core_config_dict = _extract_core_config_from_checkpoint(ckpt)

    if core_config_dict is None:
        logger.error(
            "Could not determine the model configuration. "
            "Pass --config <path-to-yaml> explicitly."
        )
        sys.exit(1)

    logger.info(
        "CoreConfig resolved (n_layers=%s, d_model=%s, vocab_size=%s)",
        core_config_dict.get("n_layers"),
        core_config_dict.get("d_model"),
        core_config_dict.get("vocab_size"),
    )

    if keep_fused_kernels:
        logger.info("Keeping fused kernels as-is from config")
        core_config_dict["label_weights_path"] = None
    else:
        logger.info("Swapping fused kernels for portable inference:")
        core_config_dict = _defuse_config(core_config_dict)

    from core.hf import CoreModelConfig, CoreModelForCausalLM

    hf_config = CoreModelConfig(core_config=core_config_dict)
    hf_model = CoreModelForCausalLM(hf_config)

    # load weights
    state_dict = _extract_state_dict(ckpt, is_lightning=is_lightning)

    missing, unexpected = hf_model.core_model.load_state_dict(state_dict, strict=False)

    # Filter out non-parameter buffers / loss_fn entries that are expected
    unexpected_real = [k for k in unexpected if not k.startswith("loss_fn.")]
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected_real:
        logger.warning("Unexpected keys: %s", unexpected_real)

    if dtype is not None:
        torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        logger.info("Casting model to %s", dtype)
        hf_model = hf_model.to(torch_dtype)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Saving HF model to %s …", out)
    hf_model.save_pretrained(out)

    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        logger.info("Saving tokenizer '%s' alongside model …", tokenizer_name)
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        tok.save_pretrained(out)

    logger.info("Done. You can now load the model with:")
    logger.info(
        '  AutoModelForCausalLM.from_pretrained("%s", trust_remote_code=True)',
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CoreModel checkpoint to HuggingFace format.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .ckpt (Lightning) or .pt (raw state-dict) file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to training YAML (with model.core_config) or bare CoreConfig YAML. "
            "If omitted the script tries to extract the config from the checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the HF model to.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer name/path to save alongside the model (e.g. 'gpt2').",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Cast model weights to this dtype before saving.",
    )
    parser.add_argument(
        "--keep-fused-kernels",
        action="store_true",
        help=(
            "Keep fused kernels (rms_fast, fused_cross_entropy, fused RoPE, flash_attn_4) "
            "in the saved config. Use when the eval machine has GPU kernel support. "
            "By default, fused kernels are swapped to portable equivalents."
        ),
    )

    args = parser.parse_args()
    convert(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        tokenizer_name=args.tokenizer,
        dtype=args.dtype,
        keep_fused_kernels=args.keep_fused_kernels,
    )


if __name__ == "__main__":
    main()
