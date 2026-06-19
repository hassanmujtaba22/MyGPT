# Running MyGPT with Ollama

[Ollama](https://ollama.com) is the easiest way to run language models locally.
There are two ways to use it with this project.

## Option A — No training, just run an open model locally

If you don't need fine-tuning (or have no GPU), you can get a great local
assistant in minutes and customize its behavior with a system prompt.

```bash
# 1. Install Ollama from https://ollama.com, then pull a model:
ollama pull qwen2.5:3b        # or: llama3.2:3b, phi3.5

# 2. Chat:
ollama run qwen2.5:3b
```

Customize personality/behavior with a **Modelfile** (no training needed):

```Dockerfile
# File: Modelfile
FROM qwen2.5:3b
SYSTEM "You are MyGPT, a concise assistant that always answers in plain English."
PARAMETER temperature 0.7
```

```bash
ollama create mygpt -f Modelfile
ollama run mygpt
```

## Option B — Run your FINE-TUNED model with Ollama

After training and merging (`scripts/merge.py`), convert the merged model to
GGUF and load it into Ollama:

```bash
# 1. Merge the adapter into a standalone model
python scripts/merge.py --config configs/default.yaml --out outputs/mygpt-merged

# 2. Convert to GGUF using llama.cpp's converter
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py outputs/mygpt-merged \
    --outfile mygpt.gguf --outtype q4_k_m

# 3. Create an Ollama model from the GGUF
cat > Modelfile <<'EOF'
FROM ./mygpt.gguf
SYSTEM "You are MyGPT, fine-tuned by me."
PARAMETER temperature 0.7
EOF

ollama create mygpt -f Modelfile
ollama run mygpt
```

Now `mygpt` runs fully locally — including through Ollama's HTTP API at
`http://localhost:11434`, which you can call from any app.
