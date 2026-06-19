#!/usr/bin/env python
"""Interactive chat with your fine-tuned MyGPT model.

Usage:
  python scripts/chat.py --config configs/default.yaml
  python scripts/chat.py --config configs/default.yaml --base   # base model only

Loads the base model and applies your trained LoRA adapter on top.
Type 'exit' or Ctrl-C to quit.
"""
import argparse
import sys
from pathlib import Path

from _common import bnb_available, cuda_available, load_config, pick_dtype


def main():
    ap = argparse.ArgumentParser(description="Chat with MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter.")
    ap.add_argument("--system", default="You are MyGPT, a helpful, concise assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)

    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")

    mcfg = cfg["model"]
    adapter_dir = cfg["training"]["output_dir"]
    dtype, _ = pick_dtype(cfg["training"].get("precision", "auto"))

    use_qlora = mcfg.get("load_in_4bit", True) and cuda_available() and bnb_available()
    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {mcfg['name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if cuda_available() else None,
    )

    if not args.base:
        if Path(adapter_dir).exists():
            from peft import PeftModel
            print(f"Applying fine-tuned adapter: {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)
        else:
            print(f"NOTE: No adapter found at {adapter_dir}. Using base model. "
                  f"(Train first with scripts/train.py, or pass --base.)")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 50)
    print("MyGPT chat — type 'exit' to quit.")
    print("=" * 50)

    history = [{"role": "system", "content": args.system}]
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"\nMyGPT: {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
