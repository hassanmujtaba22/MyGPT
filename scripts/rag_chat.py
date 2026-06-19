#!/usr/bin/env python
"""Chat with your documents using Retrieval-Augmented Generation (RAG).

For each question, MyGPT retrieves the most relevant chunks from your indexed
documents and grounds its answer in them — with page-aware citations — so it can
answer about your private data without retraining.

Build the index first:  python scripts/rag_ingest.py
Then:                   python scripts/rag_chat.py --config configs/default.yaml

Flags:
  --base          use the base model without the fine-tuned adapter
  --backend       'transformers' or 'ollama' (overrides config)
  --top-k N       how many chunks to retrieve (default from config)
  --show-sources  print which chunks were retrieved for each answer
"""
import argparse

from _common import load_config
from _llm import get_backend
from _rag import Retriever, build_context

RAG_SYSTEM = (
    "You are MyGPT, a helpful assistant. Answer the user's question using ONLY "
    "the provided context. Cite the sources you used with their bracket numbers, "
    "e.g. [1] or [2]. If the answer is not in the context, say you don't know "
    "rather than guessing. Be concise."
)


def build_grounded_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, citing sources like [1]."
    )


def main():
    ap = argparse.ArgumentParser(description="RAG chat with MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter.")
    ap.add_argument("--backend", choices=["transformers", "ollama"], default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--show-sources", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)

    rcfg = cfg.get("rag")
    if not rcfg:
        raise SystemExit("No 'rag' section in config. See configs/default.yaml.")
    top_k = args.top_k or rcfg.get("top_k", 4)

    retriever = Retriever(rcfg["index_dir"])
    print(f"Loaded index: {retriever.count} chunks "
          f"(embed model: {retriever.embed_model})")

    backend = get_backend(cfg, backend=args.backend, use_adapter=not args.base)

    print("\n" + "=" * 50)
    print(f"MyGPT RAG chat ({backend.describe()}) — ask about your documents.")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not question:
            continue

        hits = retriever.retrieve(question, top_k=top_k)
        context, sources = build_context(hits)

        if args.show_sources:
            print("\n  Retrieved:")
            for s in sources:
                print(f"    [{s['n']}] {s['score']:.3f}  {s['label']}  "
                      f"\"{s['preview'][:80]}...\"")

        history = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_grounded_prompt(question, context)},
        ]
        reply = backend.generate(history, max_new_tokens=args.max_new_tokens)
        print(f"\nMyGPT: {reply}")
        if not args.show_sources and sources:
            print("\n  Sources: " + "  ".join(
                f"[{s['n']}] {s['label']}" for s in sources))


if __name__ == "__main__":
    main()
