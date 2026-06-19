#!/usr/bin/env python
"""Merge the trained LoRA adapter into the base model to create a standalone
model directory (useful for export, sharing, or converting to GGUF/Ollama).

Usage:  python scripts/merge.py --config configs/default.yaml --out outputs/mygpt-merged
"""
import argparse
import sys
from pathlib import Path

from _common import load_config, pick_dtype


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/mygpt-merged")
    args = ap.parse_args()
    cfg = load_config(args.config)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")

    mcfg = cfg["model"]
    adapter_dir = cfg["training"]["output_dir"]
    dtype, _ = pick_dtype(cfg["training"].get("precision", "auto"))

    if not Path(adapter_dir).exists():
        sys.exit(f"No adapter at {adapter_dir}. Train first with scripts/train.py")

    print(f"Loading base model {mcfg['name']} in full precision for merge...")
    # Merge must be done WITHOUT 4-bit quantization.
    base = AutoModelForCausalLM.from_pretrained(mcfg["name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])

    print(f"Applying adapter {adapter_dir} and merging...")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    print(f"Saving merged model to {args.out}")
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("\nDone. To run with Ollama, convert this to GGUF — see docs/ollama.md")


if __name__ == "__main__":
    main()
