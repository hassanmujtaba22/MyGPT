# Coding assistant — train MyGPT on your GitHub repos

This turns MyGPT into a coding assistant that writes in **your** style by
fine-tuning a code-specialized model on **your own repositories**.

## The big picture

```
GitHub repos  ──►  github_dataset.py  ──►  data/code_train.jsonl  ──►  train.py
                   (clone + convert)        (chat-format examples)      (QLoRA)
                                                                          │
                                                          chat.py / serve.py / rag_chat.py
```

Two things to know up front:

1. **You need a GPU to train.** Fine-tuning a 1.5B–7B code model is a GPU job.
   No GPU? Use a free/cheap cloud GPU (Colab, Kaggle, RunPod, Lambda) — run the
   two commands there, download the small adapter, then run it locally. Or skip
   training: `Qwen2.5-Coder` base models are already strong coders you can run
   locally via the coding config or Ollama.
2. **Your code stays local.** Cloning and dataset-building happen on your
   machine. The generated dataset and cloned repos are git-ignored so you don't
   accidentally publish private code.

## Step 1 — Build the dataset from your repos

```bash
# A token with 'repo' scope lets it include your PRIVATE repos too.
# Create one at: https://github.com/settings/tokens
export GITHUB_TOKEN=ghp_xxxxxxxx

python scripts/github_dataset.py --user hassanmujtaba22 --out data/code_train.jsonl
```

What it does:
- Lists your repos via the GitHub API (public + private with a token).
- Shallow-clones each, walks source files, and **skips** `node_modules`,
  build output, lockfiles, minified/`.d.ts`/test files, and anything that looks
  like a secret (`.env`, keys, credentials).
- Produces chat-format examples of two kinds:
  - **completion** — given the first part of one of your files, continue it
    (learns your style across all languages: TS/JS/Python/…)
  - **py-docstring** — given a function's docstring, write the function
    (high-quality Python instruction pairs).

Handy flags:
```bash
--max-repos 5              # try a few repos first
--include-private false    # public repos only
--include-forks true       # include forks (off by default)
--max-per-repo 200         # balance: cap examples per repo
--modes completion         # only one example type
--keep-clones              # keep cloned repos in .cache/repos
```

Then sanity-check it:
```bash
python scripts/prepare_data.py --config configs/coding.yaml
```

> ⚠️ Review the dataset before training. Fine-tuning **memorizes** patterns, so
> make sure no secrets slipped in. Don't train on client code you're not
> licensed to reuse.

## Step 2 — Fine-tune

```bash
python scripts/train.py --config configs/coding.yaml
```

`configs/coding.yaml` uses a code base model (`Qwen2.5-Coder`, sized by VRAM —
run `scripts/check_hardware.py`), a longer 2048-token context, and a slightly
higher LoRA rank for code. The adapter is saved to `outputs/mygpt-coder-adapter`.

## Step 3 — Use your coding model

```bash
# CLI, with a coding system prompt
python scripts/chat.py --config configs/coding.yaml --coding

# Web UI (chat + RAG over your docs)
python scripts/serve.py --config configs/coding.yaml

# Ground answers in a repo's docs/specs too (RAG): drop files in knowledge/
python scripts/rag_ingest.py --config configs/coding.yaml
python scripts/rag_chat.py   --config configs/coding.yaml --show-sources
```

Tip: even **before** training, `--config configs/coding.yaml` loads the strong
`Qwen2.5-Coder` base model, so you get a capable coding assistant immediately;
fine-tuning then adapts it to your conventions.

## No local GPU? Train in the cloud

1. Open a notebook on Colab/Kaggle (free T4/P100) or rent a GPU on RunPod/Lambda.
2. `git clone` this repo there, `pip install -r requirements.txt`.
3. Set `GITHUB_TOKEN`, run Step 1 and Step 2.
4. Download `outputs/mygpt-coder-adapter/` (a few MB) to your machine and run
   Step 3 locally — inference is light enough for CPU/Apple Silicon, especially
   via Ollama after merging (see `docs/ollama.md`).

## How much data / how long?

- Your ~80+ repos will likely yield thousands of examples — plenty. Start with
  `--max-repos 5` to validate the flow, then run the full set.
- 2 epochs over a few thousand examples on a modern GPU is typically under an
  hour for a 1.5B–3B model.
