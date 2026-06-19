#!/usr/bin/env python
"""MyGPT web UI server — a chat interface combining your fine-tuned model with
optional RAG over your documents.

Usage:
  python scripts/serve.py --config configs/default.yaml
  python scripts/serve.py --backend ollama --port 8000
  python scripts/serve.py --base                    # base model, no adapter

Then open http://localhost:8000 in your browser.

Endpoints:
  GET  /             -> the chat UI
  GET  /api/info     -> backend + RAG status
  POST /api/chat     -> {message, history, use_rag, top_k} -> {reply, sources}
"""
import argparse
from pathlib import Path

from _common import load_config
from _llm import get_backend
from _rag import build_context

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"

RAG_SYSTEM = (
    "You are MyGPT, a helpful assistant. When context is provided, answer using "
    "ONLY that context and cite sources with their bracket numbers, e.g. [1]. If "
    "the answer isn't in the context, say you don't know. Be concise."
)
PLAIN_SYSTEM = "You are MyGPT, a helpful, concise assistant."


def create_app(cfg, backend, retriever, default_top_k):
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError:
        raise SystemExit(
            "Web UI needs FastAPI. Run: pip install -r requirements.txt"
        )

    app = FastAPI(title="MyGPT")

    class ChatRequest(BaseModel):
        message: str
        history: list = []
        use_rag: bool = False
        top_k: int = default_top_k

    @app.get("/api/info")
    def info():
        return {
            "backend": backend.describe(),
            "rag_available": retriever is not None,
            "rag_chunks": retriever.count if retriever else 0,
            "default_top_k": default_top_k,
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        sources = []
        use_rag = req.use_rag and retriever is not None

        if use_rag:
            hits = retriever.retrieve(req.message, top_k=req.top_k)
            context, sources = build_context(hits)
            system = RAG_SYSTEM
            user_content = (
                f"Context:\n{context}\n\nQuestion: {req.message}\n\n"
                f"Answer using only the context above, citing sources like [1]."
            )
        else:
            system = PLAIN_SYSTEM
            user_content = req.message

        # Rebuild the message list: system + prior turns + this (grounded) turn.
        messages = [{"role": "system", "content": system}]
        for turn in req.history:
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_content})

        reply = backend.generate(messages, max_new_tokens=512)
        return JSONResponse({"reply": reply, "sources": sources})

    @app.get("/")
    def index():
        return FileResponse(str(WEBUI_DIR / "index.html"))

    if WEBUI_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEBUI_DIR)), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description="Serve the MyGPT web UI.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", action="store_true",
                    help="Use the base model without the fine-tuned adapter.")
    ap.add_argument("--backend", choices=["transformers", "ollama"], default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Load the generation backend.
    backend = get_backend(cfg, backend=args.backend, use_adapter=not args.base)

    # Load the RAG retriever if an index exists (optional).
    retriever, default_top_k = None, cfg.get("rag", {}).get("top_k", 4)
    rcfg = cfg.get("rag")
    if rcfg and Path(rcfg["index_dir"], "embeddings.npy").exists():
        from _rag import Retriever
        retriever = Retriever(rcfg["index_dir"])
        print(f"RAG enabled: {retriever.count} chunks from {rcfg['index_dir']}")
    else:
        print("RAG index not found — UI will run chat-only. "
              "Build one with scripts/rag_ingest.py to enable the RAG toggle.")

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Web UI needs uvicorn. Run: pip install -r requirements.txt")

    app = create_app(cfg, backend, retriever, default_top_k)
    print(f"\nMyGPT web UI → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
