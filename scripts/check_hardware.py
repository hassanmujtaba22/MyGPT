#!/usr/bin/env python
"""Detect this machine's hardware and recommend a fine-tuning setup.

Run this first:  python scripts/check_hardware.py
"""
import platform
import shutil
import sys


def human_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.1f} GB"


def detect_gpu():
    """Return (kind, name, vram_gb) where kind is 'cuda', 'mps', or None."""
    try:
        import torch
    except ImportError:
        return None, "torch not installed", None

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "cuda", name, vram

    # Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Silicon (Metal/MPS)", None

    return None, "No GPU detected", None


def recommend(kind, vram_gb):
    """Return a (headline, model, notes) recommendation."""
    if kind == "cuda":
        if vram_gb is None:
            vram_gb = 0
        if vram_gb >= 22:
            return (
                "Excellent — you can fine-tune 7–8B models with QLoRA.",
                "meta-llama/Llama-3.1-8B-Instruct",
                "Use 4-bit QLoRA. Batch size 1–2 with gradient accumulation.",
            )
        if vram_gb >= 12:
            return (
                "Good — fine-tune 3B models comfortably with QLoRA.",
                "Qwen/Qwen2.5-3B-Instruct",
                "Use 4-bit QLoRA, max_seq_len ~1024–2048.",
            )
        if vram_gb >= 6:
            return (
                "Workable — stick to ~1B models with QLoRA.",
                "meta-llama/Llama-3.2-1B-Instruct",
                "Use 4-bit QLoRA, small batch + gradient accumulation, "
                "max_seq_len ~512–1024.",
            )
        return (
            "Limited VRAM — try the smallest models, keep sequences short.",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "Reduce max_seq_len to 512 and lora_r to 8 if you hit OOM.",
        )

    if kind == "mps":
        return (
            "Apple Silicon detected. bitsandbytes (4-bit) is NOT supported here.\n"
            "  Best path: use Apple's MLX framework for local fine-tuning,\n"
            "  or run this project's LoRA (non-quantized) on small models only.",
            "mlx-community/Llama-3.2-1B-Instruct-4bit (via MLX)",
            "See docs/faq.md for the MLX route. For inference, Ollama works great.",
        )

    # CPU only
    return (
        "No GPU found. Fine-tuning on CPU is impractically slow.\n"
        "  Recommended: rent a GPU (Colab/Kaggle/RunPod) to TRAIN, then run\n"
        "  inference locally — OR skip training and use Ollama with a custom\n"
        "  system prompt. See docs/ollama.md.",
        "Qwen/Qwen2.5-3B-Instruct (for local INFERENCE via Ollama)",
        "If you must train on CPU, use SmolLM2-135M and expect it to be slow.",
    )


def main():
    print("=" * 64)
    print("  MyGPT — Hardware Check")
    print("=" * 64)

    # System info
    print(f"\nOS:            {platform.system()} {platform.release()}")
    print(f"Python:        {platform.python_version()}")
    print(f"CPU cores:     {shutil.os.cpu_count()}")

    # RAM
    try:
        import psutil  # optional
        ram = psutil.virtual_memory().total
        print(f"System RAM:    {human_gb(ram)}")
    except ImportError:
        print("System RAM:    (install 'psutil' to see this)")

    # Torch / GPU
    try:
        import torch
        print(f"PyTorch:       {torch.__version__}")
    except ImportError:
        print("PyTorch:       NOT INSTALLED — run: pip install -r requirements.txt")

    kind, name, vram = detect_gpu()
    print(f"\nGPU:           {name}")
    if vram is not None:
        print(f"VRAM:          {vram:.1f} GB")

    # bitsandbytes availability (QLoRA)
    try:
        import bitsandbytes  # noqa: F401
        print("QLoRA (4-bit): available (bitsandbytes installed)")
    except Exception:
        print("QLoRA (4-bit): NOT available on this machine")

    headline, model, notes = recommend(kind, vram)
    print("\n" + "-" * 64)
    print("RECOMMENDATION")
    print("-" * 64)
    print(headline)
    print(f"\n  Suggested base model:  {model}")
    print(f"  Notes:                 {notes}")
    print("\nNext: set this model in configs/default.yaml, add your examples to")
    print("data/train.jsonl, then run:  python scripts/train.py\n")


if __name__ == "__main__":
    sys.exit(main())
