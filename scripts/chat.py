#!/usr/bin/env python
"""Interactive chat with your fine-tuned MyGPT model.

Usage:
  python scripts/chat.py --config configs/default.yaml
  python scripts/chat.py --config configs/default.yaml --base   # base model only

Loads the base model and applies your trained LoRA adapter on top.
Type 'exit' or Ctrl-C to quit.
"""
import argparse

from _common import generate_reply, load_config, load_model_and_tokenizer


def main():
    ap = argparse.ArgumentParser(description="Chat with MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter.")
    ap.add_argument("--system", default="You are MyGPT, a helpful, concise assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model, tokenizer = load_model_and_tokenizer(cfg, use_adapter=not args.base)

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
        reply = generate_reply(model, tokenizer, history,
                               max_new_tokens=args.max_new_tokens)
        print(f"\nMyGPT: {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
