# FAQ & Troubleshooting

### "CUDA out of memory"
Lower memory use, in roughly this order:
1. Reduce `model.max_seq_len` (e.g. 1024 → 512).
2. Keep `training.batch_size: 1` and raise `training.grad_accum` (e.g. 8 → 16).
3. Use a smaller base model (see `check_hardware.py`).
4. Lower `lora.r` (e.g. 16 → 8).

### "bitsandbytes" won't install / 4-bit not available
`bitsandbytes` needs an NVIDIA GPU + CUDA. On Apple Silicon or CPU it's not
available; the scripts auto-fall back to plain LoRA. For real fine-tuning
without an NVIDIA GPU, rent one (Colab/Kaggle/RunPod) or use the MLX route below.

### Apple Silicon (M1/M2/M3/M4)
For local fine-tuning on a Mac, Apple's **MLX** framework is the best path:
```bash
pip install mlx-lm
mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train --data ./data \
  --iters 300
```
MLX expects its own data layout (a `data/` folder with `train.jsonl` /
`valid.jsonl` using `{"text": ...}` or chat format). See the mlx-lm docs.
For **inference**, Ollama works great on Mac out of the box (see `ollama.md`).

### The model isn't learning my style
- Add more, higher-quality examples (aim for a few hundred).
- Increase `training.epochs` (e.g. 3 → 5) — but watch for overfitting.
- Make sure your examples are consistent in format and tone.
- Verify data with `python scripts/prepare_data.py`.

### It overfit (repeats training data, ignores new questions)
- Reduce `epochs`, reduce `learning_rate`, or add more varied data.
- Add an `eval_file` and watch eval loss stop improving.

### "Access to model is gated"
Some models (Llama) require accepting a license on Hugging Face, then:
```bash
huggingface-cli login
```
Or switch `model.name` to a fully-open model like `Qwen/Qwen2.5-3B-Instruct`.

### How long does training take?
Depends on model size, data size, and GPU. A 1–3B model on a few hundred
examples for 3 epochs is typically minutes to an hour on a modern GPU.

### Can I train on CPU?
Technically yes with a tiny model (e.g. SmolLM2-135M), but it's slow and not
recommended. Rent a GPU for an hour instead — it's cheap and far faster.
