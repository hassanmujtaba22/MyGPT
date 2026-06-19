#!/usr/bin/env python
"""Interactive chat with your MyGPT model.

Usage:
  python scripts/chat.py --config configs/default.yaml
  python scripts/chat.py --base                       # base model, no adapter
  python scripts/chat.py --backend ollama             # generate via Ollama

The backend (local transformers vs Ollama) comes from the config 'backend'
section and can be overridden with --backend. Type 'exit' or Ctrl-C to quit.
"""
import argparse

from _common import load_config
from _llm import get_backend


def main():
    ap = argparse.ArgumentParser(description="Chat with MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter "
                         "(transformers backend only).")
    ap.add_argument("--backend", choices=["transformers", "ollama"], default=None,
                    help="Override the generation backend from the config.")
    ap.add_argument("--system", default="You are MyGPT, a helpful, concise assistant.")
    ap.add_argument("--coding", action="store_true",
                    help="Use a coding-assistant system prompt.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)

    system = args.system
    if args.coding:
        system = ("You are MyGPT, an expert pair programmer fine-tuned on the "
                  "user's own code. Write clean, idiomatic code in their style, "
                  "explain your reasoning briefly, and prefer working examples.")

    backend = get_backend(cfg, backend=args.backend, use_adapter=not args.base)

    print("\n" + "=" * 50)
    print(f"MyGPT chat ({backend.describe()}) — type 'exit' to quit.")
    print("=" * 50)

    history = [{"role": "system", "content": system}]
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
        reply = backend.generate(history, max_new_tokens=args.max_new_tokens)
        print(f"\nMyGPT: {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
