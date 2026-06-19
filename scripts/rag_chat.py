#!/usr/bin/env python
"""Chat with your documents using Retrieval-Augmented Generation (RAG).

For each question, MyGPT retrieves the most relevant chunks from your indexed
documents and grounds its answer in them — so it can answer about your private
data without retraining.

Build the index first:  python scripts/rag_ingest.py
Then:                   python scripts/rag_chat.py --config configs/default.yaml

Flags:
  --base          use the base model without the fine-tuned adapter
  --top-k N       how many chunks to retrieve (default from config)
  --show-sources  print which chunks were retrieved for each answer
"""
import argparse

from _common import generate_reply, load_config, load_model_and_tokenizer
from _rag import embed_texts, get_embedder, load_index, search

RAG_SYSTEM = (
    "You are MyGPT, a helpful assistant. Answer the user's question using ONLY "
    "the provided context. If the answer is not in the context, say you don't "
    "know rather than guessing. Be concise and cite sources when relevant."
)


def build_grounded_prompt(question: str, chunks: list) -> str:
    """Combine retrieved chunks and the question into a single user turn."""
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source", "?")
        context_blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def main():
    ap = argparse.ArgumentParser(description="RAG chat with MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter.")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--show-sources", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)

    rcfg = cfg.get("rag")
    if not rcfg:
        raise SystemExit("No 'rag' section in config. See configs/default.yaml.")
    top_k = args.top_k or rcfg.get("top_k", 4)

    # Load the index, then the embedder that built it (kept consistent via meta).
    embeddings, records, meta = load_index(rcfg["index_dir"])
    print(f"Loaded index: {meta['count']} chunks "
          f"(embed model: {meta['embed_model']})")
    embedder = get_embedder(meta["embed_model"])

    # Load the generation model (fine-tuned adapter applied if present).
    model, tokenizer = load_model_and_tokenizer(cfg, use_adapter=not args.base)

    print("\n" + "=" * 50)
    print("MyGPT RAG chat — ask about your documents. Type 'exit' to quit.")
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

        # Retrieve
        qvec = embed_texts(embedder, [question])[0]
        hits = search(qvec, embeddings, records, top_k=top_k)

        if args.show_sources:
            print("\n  Retrieved:")
            for i, h in enumerate(hits, start=1):
                preview = h["text"][:80].replace("\n", " ")
                print(f"    [{i}] {h['score']:.3f}  {h['source']}  \"{preview}...\"")

        # Generate (fresh single-turn grounding each question for reliability)
        grounded = build_grounded_prompt(question, hits)
        history = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": grounded},
        ]
        reply = generate_reply(model, tokenizer, history,
                               max_new_tokens=args.max_new_tokens)
        print(f"\nMyGPT: {reply}")


if __name__ == "__main__":
    main()
