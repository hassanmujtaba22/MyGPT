# MyGPT Web UI

A local web chat interface that combines your fine-tuned model with optional
RAG over your documents — including page-aware source citations.

## Run it

```bash
pip install -r requirements.txt
python scripts/serve.py --config configs/default.yaml
# open http://localhost:8000
```

Useful flags:

```bash
python scripts/serve.py --backend ollama       # generate via Ollama instead
python scripts/serve.py --base                 # base model, ignore the adapter
python scripts/serve.py --host 0.0.0.0 --port 8080   # expose on your network
```

## Features

- **Chat** with your fine-tuned MyGPT model.
- **"Use my documents (RAG)" toggle** — when on, each answer is grounded in the
  most relevant chunks from your indexed files.
- **Citations** — answers reference `[1]`, `[2]`, and a Sources panel lists each
  one with its file and **page number** (for PDFs) plus a relevance score.
- **Backend indicator** in the header shows whether you're on transformers or
  Ollama, and which model.
- The RAG toggle auto-disables if no index exists; build one with
  `scripts/rag_ingest.py`.

## How it works

`scripts/serve.py` is a small FastAPI app:

- `GET /` serves `webui/index.html` (a self-contained page — no build step).
- `GET /api/info` reports the backend and RAG status.
- `POST /api/chat` takes `{message, history, use_rag, top_k}`. When `use_rag` is
  true it retrieves chunks, builds a grounded prompt, and returns the reply plus
  a `sources` list for the UI.

Generation goes through the same backend abstraction as the CLI (`_llm.py`), so
the UI works identically whether you use transformers or Ollama.

## Exposing beyond localhost

`--host 0.0.0.0` makes it reachable on your LAN. There's no authentication, so
only do this on a trusted network. For remote access, put it behind a reverse
proxy (nginx/Caddy) with TLS and auth.
