#!/usr/bin/env python
"""Validate and inspect your training data before fine-tuning.

Usage:  python scripts/prepare_data.py --config configs/default.yaml
"""
import argparse

from _common import load_config, read_jsonl, validate_messages


def main():
    ap = argparse.ArgumentParser(description="Validate MyGPT training data.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_file = cfg["data"]["train_file"]

    print(f"Reading {train_file} ...")
    records = read_jsonl(train_file)
    validate_messages(records)
    print(f"OK: {len(records)} examples, all well-formed.\n")

    # Token-length stats (needs the tokenizer; falls back to whitespace counts).
    lengths = []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        for rec in records:
            text = tok.apply_chat_template(rec["messages"], tokenize=False)
            lengths.append(len(tok(text)["input_ids"]))
        unit = "tokens"
    except Exception as e:
        print(f"(Tokenizer unavailable — {e}. Using word counts instead.)")
        for rec in records:
            text = " ".join(m["content"] for m in rec["messages"])
            lengths.append(len(text.split()))
        unit = "words"

    if lengths:
        lengths.sort()
        n = len(lengths)
        print(f"Length stats ({unit} per example):")
        print(f"  min:    {lengths[0]}")
        print(f"  median: {lengths[n // 2]}")
        print(f"  max:    {lengths[-1]}")
        max_seq = cfg["model"]["max_seq_len"]
        over = sum(1 for l in lengths if l > max_seq)
        if over:
            print(f"\n  WARNING: {over} example(s) exceed max_seq_len={max_seq} "
                  f"and will be truncated. Consider raising max_seq_len or "
                  f"splitting those examples.")
    print("\nData looks ready. Run: python scripts/train.py")


if __name__ == "__main__":
    main()
