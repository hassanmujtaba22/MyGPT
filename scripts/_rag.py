"""RAG building blocks for MyGPT: document loading, chunking, embeddings, and a
lightweight on-disk vector store.

The store uses NumPy cosine similarity by default (no native deps beyond
sentence-transformers) which is plenty for thousands of chunks. For very large
corpora, swap in FAISS — see docs/rag.md.
"""
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_documents(source_dir: str) -> list:
    """Load all supported documents under source_dir into {path, text} dicts.

    Supports .txt/.md/.rst natively, and .pdf if 'pypdf' is installed.
    """
    src = Path(source_dir)
    if not src.exists():
        sys.exit(f"Knowledge directory not found: {source_dir}")

    docs = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        if text and text.strip():
            docs.append({"path": str(path), "text": text})
    if not docs:
        sys.exit(f"No readable documents found in {source_dir} "
                 f"(supported: .txt, .md, .rst, .pdf).")
    return docs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  Skipping {path.name}: install 'pypdf' to read PDFs.")
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        print(f"  Could not read {path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Split text into overlapping word-based chunks.

    Word-based chunking keeps things dependency-free and predictable. Overlap
    preserves context that would otherwise be cut at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4
    chunks, start = [], 0
    step = chunk_size - overlap
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


def build_chunks(docs: list, chunk_size: int, overlap: int) -> list:
    """Turn documents into a flat list of chunk records with source metadata."""
    records = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            records.append({
                "text": chunk,
                "source": doc["path"],
                "chunk": i,
            })
    return records


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedder(model_name: str):
    """Load a sentence-transformers embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is required for RAG.\n"
                 "Run: pip install -r requirements.txt")
    return SentenceTransformer(model_name)


def embed_texts(embedder, texts: list):
    """Embed a list of texts into a normalized float32 NumPy array."""
    import numpy as np
    vecs = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Vector store (save / load / search)
# ---------------------------------------------------------------------------

def save_index(index_dir: str, embeddings, records: list, embed_model: str) -> None:
    """Persist embeddings (.npy) and chunk metadata (.jsonl) to index_dir."""
    import numpy as np
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(
        json.dumps({"embed_model": embed_model, "count": len(records)}, indent=2)
    )


def load_index(index_dir: str):
    """Load an index built by save_index. Returns (embeddings, records, meta)."""
    import numpy as np
    idx = Path(index_dir)
    emb_path = idx / "embeddings.npy"
    if not emb_path.exists():
        sys.exit(f"No index found at {index_dir}. Build it first:\n"
                 f"  python scripts/rag_ingest.py")
    embeddings = np.load(emb_path)
    records = []
    with (idx / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    meta = json.loads((idx / "meta.json").read_text())
    return embeddings, records, meta


def search(query_vec, embeddings, records: list, top_k: int = 4) -> list:
    """Return the top_k most similar chunks with scores (cosine similarity).

    Embeddings are normalized, so a single matrix-vector dot product gives
    cosine similarity for the whole store at once.
    """
    import numpy as np
    scores = embeddings @ query_vec  # (N,)
    k = min(top_k, len(records))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    results = []
    for i in top_idx:
        rec = dict(records[int(i)])
        rec["score"] = float(scores[int(i)])
        results.append(rec)
    return results
