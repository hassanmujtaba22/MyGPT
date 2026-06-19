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


def load_model_and_tokenizer(cfg: dict, use_adapter: bool = True):
    """Load the base model (4-bit QLoRA when possible) plus tokenizer, and
    apply the fine-tuned LoRA adapter if one exists. Shared by chat.py and
    rag_chat.py so there's a single loading code path.
    """
    import sys

    try:
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")

    mcfg = cfg["model"]
    adapter_dir = cfg["training"]["output_dir"]
    dtype, _ = pick_dtype(cfg["training"].get("precision", "auto"))

    use_qlora = mcfg.get("load_in_4bit", True) and cuda_available() and bnb_available()
    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {mcfg['name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if cuda_available() else None,
    )

    if use_adapter and Path(adapter_dir).exists():
        from peft import PeftModel
        print(f"Applying fine-tuned adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
    elif use_adapter:
        print(f"NOTE: No adapter found at {adapter_dir}. Using base model. "
              f"(Train first with scripts/train.py, or pass --base.)")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_reply(model, tokenizer, history: list, max_new_tokens: int = 512,
                   temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Run one generation pass over a chat history and return the reply text."""
    import torch
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
