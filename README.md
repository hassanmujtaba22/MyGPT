# MyGPT — Fine-tune your own AI model, locally

This project lets you **fine-tune an existing open-source language model** on your
own data and then run it on your machine. It uses **QLoRA** (4-bit quantized
LoRA), the modern, memory-efficient technique that makes fine-tuning possible
even on consumer hardware. It also includes **RAG** so MyGPT can answer
questions grounded in your own documents.

You don't train a model from scratch (that costs millions). Instead you take a
strong open base model (e.g. Llama 3.2, Qwen2.5, Phi-3) and teach it your
style, knowledge, or task with a small dataset.

**Two complementary techniques** (use either or both):
- **Fine-tuning (QLoRA)** — teaches the model your *style and behavior*.
- **RAG** — supplies the model with *facts* from your documents at query time,
  no retraining needed. See [`docs/rag.md`](docs/rag.md).

---

## TL;DR — the 4 steps

```bash
# 1. Install dependencies (in a virtual environment)
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Check what your hardware can handle — tells you which model/config to use
python scripts/check_hardware.py

# 3. Prepare your data (edit data/train.jsonl with your own examples)
#    A sample dataset is already provided.

# 4. Fine-tune, then chat with your model
python scripts/train.py   --config configs/default.yaml
python scripts/chat.py    --config configs/default.yaml
```

### Optional: chat with your own documents (RAG)

```bash
# Put your files (.txt/.md/.rst/.pdf) in the knowledge/ folder, then:
python scripts/rag_ingest.py --config configs/default.yaml   # build the index
python scripts/rag_chat.py   --config configs/default.yaml --show-sources
```
RAG needs no GPU or training and works with the base or fine-tuned model. Full
guide in [`docs/rag.md`](docs/rag.md).

---

## How fine-tuning works (the short version)

1. **Base model** — a pretrained LLM that already understands language.
2. **Your dataset** — examples of the inputs and the outputs you want
   (instruction → response pairs, conversations, your writing, Q&A about your
   docs, etc.).
3. **LoRA adapters** — instead of updating all billions of weights, we train
   tiny "adapter" matrices (a few million params). This is fast and fits in
   little memory.
4. **QLoRA** — loads the base model in 4-bit precision so even a big model fits
   on a small GPU, while the adapters stay full precision.
5. **Result** — a small adapter file you can merge into the base model or load
   on top of it at inference time.

---

## Choosing hardware & model

Run `python scripts/check_hardware.py` first. General guidance:

| Hardware | What you can do | Suggested base model |
|---|---|---|
| **NVIDIA GPU, 24GB+ VRAM** | Fine-tune 7–8B models comfortably | `meta-llama/Llama-3.1-8B-Instruct` |
| **NVIDIA GPU, 8–16GB VRAM** | Fine-tune 1–3B with QLoRA | `Qwen/Qwen2.5-3B-Instruct` |
| **NVIDIA GPU, 6–8GB VRAM** | Fine-tune ~1B with QLoRA | `meta-llama/Llama-3.2-1B-Instruct` |
| **Apple Silicon (M-series)** | Fine-tune small models via MLX (see notes) | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| **CPU only** | Fine-tuning is impractical | Run inference only (see "CPU only" below) |

> **No GPU?** Real fine-tuning on CPU is too slow to be useful. Two good options:
> 1. **Rent a GPU** for an hour or two (Google Colab, Kaggle, RunPod, Lambda,
>    Vast.ai). Run `train.py` there, download the adapter, run inference locally.
> 2. **Skip fine-tuning** and just run a strong open model locally with
>    [Ollama](https://ollama.com). Customize behavior with a system prompt /
>    Modelfile instead of training. See `docs/ollama.md`.

---

## Project layout

```
MyGPT/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # model, data, and training settings
├── data/
│   ├── train.jsonl           # your training examples (edit this!)
│   └── README.md             # data format guide
├── knowledge/                # drop RAG source docs here (.txt/.md/.rst/.pdf)
│   └── example.md            # sample doc to try RAG immediately
├── scripts/
│   ├── check_hardware.py     # detect GPU/RAM and recommend a setup
│   ├── prepare_data.py       # validate & format your dataset
│   ├── train.py              # QLoRA fine-tuning
│   ├── merge.py              # merge adapter into base model (optional)
│   ├── chat.py               # chat with your fine-tuned model
│   ├── rag_ingest.py         # build a RAG index from knowledge/
│   ├── rag_chat.py           # chat grounded in your documents (RAG)
│   ├── _rag.py               # RAG internals (chunking, embeddings, search)
│   └── _common.py            # shared helpers (config, model loading)
└── docs/
    ├── ollama.md             # run/export with Ollama (no-training path)
    ├── rag.md                # retrieval-augmented generation guide
    └── faq.md
```

---

## Getting a base model

Most models on Hugging Face are open but some (like Llama) are **gated** — you
must accept the license once on the model page, then log in:

```bash
pip install huggingface_hub
huggingface-cli login        # paste a token from https://huggingface.co/settings/tokens
```

Fully open alternatives that need no gating: `Qwen/Qwen2.5-3B-Instruct`,
`microsoft/Phi-3.5-mini-instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

---

## License & responsibility

Respect each base model's license and use your fine-tuned model responsibly.
Your training data is yours — keep private data private.
