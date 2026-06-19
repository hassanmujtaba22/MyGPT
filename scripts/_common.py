"""Shared helpers for MyGPT scripts."""
import json
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required. Run: pip install -r requirements.txt")
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> list:
    """Read a .jsonl file into a list of dicts, with friendly error messages."""
    records = []
    p = Path(path)
    if not p.exists():
        sys.exit(f"Data file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"Invalid JSON on line {lineno} of {path}: {e}")
    return records


def validate_messages(records: list) -> None:
    """Ensure each record has a valid 'messages' list."""
    valid_roles = {"system", "user", "assistant"}
    for i, rec in enumerate(records, start=1):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            sys.exit(f"Record {i}: missing or empty 'messages' list.")
        roles = set()
        for m in msgs:
            role = m.get("role")
            if role not in valid_roles:
                sys.exit(f"Record {i}: invalid role {role!r}.")
            if not isinstance(m.get("content"), str):
                sys.exit(f"Record {i}: 'content' must be a string.")
            roles.add(role)
        if "user" not in roles or "assistant" not in roles:
            sys.exit(f"Record {i}: needs at least one user and one assistant turn.")


def pick_dtype(precision: str):
    """Resolve precision string to a torch dtype + flag."""
    import torch
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        if torch.cuda.is_available():
            return torch.float16, "fp16"
        return torch.float32, "fp32"
    return {
        "bf16": (torch.bfloat16, "bf16"),
        "fp16": (torch.float16, "fp16"),
        "fp32": (torch.float32, "fp32"),
    }.get(precision, (torch.float32, "fp32"))


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False
