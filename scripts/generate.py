from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

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
        if isinstance(path_key, tuple):
            section_path, field = path_key
        else:
            section_path, field = path_key.rsplit(".", 1)

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


def load_from_checkpoint(checkpoint_path: str, device: str = "cpu", keep_fused: bool = False):
    """Load a CoreModelForCausalLM from a Lightning .ckpt or raw .pt file."""
    from core.hf import CoreModelConfig, CoreModelForCausalLM

    ckpt_path = Path(checkpoint_path)
    is_lightning = ckpt_path.suffix == ".ckpt"

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hp = ckpt.get("hyper_parameters") or ckpt.get("hparams")
    if hp is None or "config" not in hp:
        logger.error(
            "Checkpoint has no hyper_parameters.config. "
            "Use --hf-model with a pre-converted directory instead, "
            "or convert first with scripts/convert_to_hf.py."
        )
        sys.exit(1)

    config_obj = hp["config"]
    cfg_dict = config_obj.model_dump() if hasattr(config_obj, "model_dump") else dict(config_obj)

    if keep_fused:
        logger.info("Keeping fused kernels as-is from checkpoint config")
        cfg_dict["label_weights_path"] = None
    else:
        logger.info("Swapping fused kernels for portable inference:")
        cfg_dict = _defuse_config(cfg_dict)

    hf_config = CoreModelConfig(core_config=cfg_dict)
    model = CoreModelForCausalLM(hf_config)

    sd = ckpt.get("state_dict", ckpt)
    if is_lightning:
        sd = {k.removeprefix("model."): v for k, v in sd.items()}

    missing, unexpected = model.core_model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("loss_fn.")]
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    logger.info("Loaded %s parameters", f"{model.core_model.num_parameters():,}")
    return model.to(device).eval()


def load_from_hf(hf_model_path: str, device: str = "cpu"):
    """Load from a pre-converted HF directory."""
    from core.hf import CoreModelForCausalLM

    logger.info("Loading HF model: %s", hf_model_path)
    model = CoreModelForCausalLM.from_pretrained(hf_model_path)
    logger.info("Loaded %s parameters", f"{model.core_model.num_parameters():,}")
    return model.to(device).eval()


def generate(
    model,
    prompts: List[str],
    tokenizer_name: str = "gpt2",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> List[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = next(model.parameters()).device
    outputs = []
    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("Using bf16 mixed precision (torch.autocast)")

    for prompt in prompts:
        input_ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        outputs.append(text)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a CoreModel.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to Lightning .ckpt or raw .pt checkpoint.")
    source.add_argument("--hf-model", help="Path to pre-converted HF model directory.")

    parser.add_argument("--prompt", nargs="+", required=True, help="One or more prompts to complete.")
    parser.add_argument("--tokenizer", default="gpt2", help="HF tokenizer name/path (default: gpt2).")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate (default: 100).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8).")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p (default: 0.95).")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1).")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--keep-fused-kernels", action="store_true",
                        help="Keep fused kernels (rms_fast, fused_cross_entropy, fused RoPE, flash_attn_4) "
                             "from the checkpoint config instead of swapping to portable equivalents. "
                             "Requires a GPU with the necessary kernel support.")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Using device: %s", device)

    if args.checkpoint:
        model = load_from_checkpoint(args.checkpoint, device=device, keep_fused=args.keep_fused_kernels)
    else:
        model = load_from_hf(args.hf_model, device=device)

    results = generate(
        model,
        prompts=args.prompt,
        tokenizer_name=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
    )

    print()
    for prompt, output in zip(args.prompt, results):
        print(f"Prompt:  {prompt}")
        print(f"Output:  {output}")
        print()


if __name__ == "__main__":
    main()
