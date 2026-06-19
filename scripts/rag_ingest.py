#!/usr/bin/env python
"""Build a RAG index from your documents.

Reads every supported file under the knowledge directory, splits them into
overlapping chunks, embeds the chunks, and saves a searchable index.

Usage:
  python scripts/rag_ingest.py --config configs/default.yaml
"""
import argparse

from _common import load_config
from _rag import (build_chunks, embed_texts, get_embedder, load_documents,
                  save_index)


def main():
    ap = argparse.ArgumentParser(description="Build the MyGPT RAG index.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    rcfg = cfg.get("rag")
    if not rcfg:
        raise SystemExit("No 'rag' section in config. See configs/default.yaml.")

    print(f"Loading documents from: {rcfg['source_dir']}")
    docs = load_documents(rcfg["source_dir"])
    print(f"  {len(docs)} document(s) found.")

    records = build_chunks(docs, rcfg["chunk_size"], rcfg["chunk_overlap"])
    print(f"  Split into {len(records)} chunk(s).")

    print(f"Loading embedding model: {rcfg['embed_model']}")
    embedder = get_embedder(rcfg["embed_model"])

    print("Embedding chunks...")
    embeddings = embed_texts(embedder, [r["text"] for r in records])

    save_index(rcfg["index_dir"], embeddings, records, rcfg["embed_model"])
    print(f"\nIndex saved to: {rcfg['index_dir']}")
    print("Now chat with your documents:")
    print(f"  python scripts/rag_chat.py --config {args.config}")


if __name__ == "__main__":
    main()
