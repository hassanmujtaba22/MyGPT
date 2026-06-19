#!/usr/bin/env python
"""Fine-tune a base model on your data with (Q)LoRA.

Usage:  python scripts/train.py --config configs/default.yaml

- On an NVIDIA GPU with bitsandbytes, uses 4-bit QLoRA (memory-efficient).
- On Apple Silicon / CPU, falls back to plain LoRA (needs more memory; use a
  small model). See README for the recommended path on each platform.
"""
import argparse
import sys

from _common import bnb_available, cuda_available, load_config, pick_dtype


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning for MyGPT.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Imports here so --help works without the heavy stack installed.
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")

    mcfg, dcfg, lcfg, tcfg = cfg["model"], cfg["data"], cfg["lora"], cfg["training"]
    dtype, prec_flag = pick_dtype(tcfg.get("precision", "auto"))

    use_qlora = mcfg.get("load_in_4bit", True) and cuda_available() and bnb_available()
    if mcfg.get("load_in_4bit", True) and not use_qlora:
        print("NOTE: 4-bit QLoRA requested but unavailable on this machine "
              "(needs NVIDIA GPU + bitsandbytes). Falling back to plain LoRA.")

    print(f"Loading base model: {mcfg['name']}  (precision: {prec_flag}, "
          f"qlora: {use_qlora})")

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if cuda_available() else None,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # LoRA adapter config.
    target = lcfg.get("target_modules", "all-linear")
    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Dataset (chat format). TRL applies the chat template automatically.
    data_files = {"train": dcfg["train_file"]}
    if dcfg.get("eval_file"):
        data_files["eval"] = dcfg["eval_file"]
    ds = load_dataset("json", data_files=data_files)

    sft_config = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        seed=tcfg["seed"],
        max_seq_length=mcfg["max_seq_len"],
        gradient_checkpointing=True,
        bf16=(prec_flag == "bf16"),
        fp16=(prec_flag == "fp16"),
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\nStarting training...\n" + "=" * 50)
    trainer.train()

    print("\nSaving adapter to:", tcfg["output_dir"])
    trainer.save_model(tcfg["output_dir"])
    tokenizer.save_pretrained(tcfg["output_dir"])
    print("\nDone! Chat with your model:")
    print(f"  python scripts/chat.py --config {args.config}")


if __name__ == "__main__":
    main()
