# RAG — chat with your own documents

RAG (Retrieval-Augmented Generation) lets MyGPT answer questions using **your
documents** without retraining. At query time it:

1. **Embeds** your question into a vector.
2. **Retrieves** the most similar chunks from your indexed documents.
3. **Grounds** the model's answer in those chunks (injected into the prompt).

This means you can update knowledge instantly (just re-index) and the model can
cite where answers came from.

## Fine-tuning vs RAG — use both

| | Fine-tuning (QLoRA) | RAG |
|---|---|---|
| Teaches | Style, tone, behavior, tasks | Facts / knowledge |
| Update cost | Retrain (needs GPU) | Re-run ingestion (seconds, CPU) |
| Best for | "Sound like X", "always format as Y" | "Answer about my docs/notes/manuals" |

They combine: a fine-tuned MyGPT that *also* retrieves from your documents.

## Quick start

```bash
# 1. Install deps (sentence-transformers etc. are in requirements.txt)
pip install -r requirements.txt

# 2. Drop your files into the knowledge/ folder (.txt, .md, .rst, .pdf)
#    A sample example.md is already there.

# 3. Build the index
python scripts/rag_ingest.py --config configs/default.yaml

# 4. Chat with your documents
python scripts/rag_chat.py --config configs/default.yaml --show-sources
```

Try the bundled sample: ask **"What is the capital of Examplestan?"** — a working
setup answers "Demoville" straight from `knowledge/example.md`.

## How it's built (no heavy infra)

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — small, fast, runs
  locally on CPU. Change it via `rag.embed_model` in the config.
- **Vector store:** a NumPy array of normalized embeddings + a JSONL of chunk
  metadata, saved under `outputs/rag_index/`. Search is a single matrix-vector
  dot product (cosine similarity). Plenty fast for thousands of chunks.
- **Chunking:** overlapping word windows (`chunk_size` / `chunk_overlap` in the
  config). Smaller chunks = more precise retrieval; larger = more context each.

## Tuning retrieval

- **Answers miss relevant info?** Increase `rag.top_k` (e.g. 4 → 6) or lower
  `chunk_size` for finer-grained chunks, then re-ingest.
- **Answers include irrelevant junk?** Lower `top_k`, or raise `chunk_size`.
- **Use `--show-sources`** to see exactly which chunks were retrieved and their
  similarity scores — the fastest way to debug quality.

## Scaling up (optional FAISS)

The NumPy store is fine into the tens of thousands of chunks. Beyond that,
install `faiss-cpu` (uncomment it in `requirements.txt`) and replace the
`search()` call in `scripts/_rag.py` with a FAISS index for sub-linear search.

## PDF support

PDFs are read via `pypdf` (included). Scanned/image PDFs need OCR first
(e.g. `ocrmypdf`) since there's no selectable text to extract.

## Privacy

Everything runs locally — embedding, indexing, retrieval, and generation. Your
documents never leave your machine.
