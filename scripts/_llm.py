"""Pluggable text-generation backends for MyGPT.

Two backends share one interface so chat.py, rag_chat.py and the web server can
generate the same way regardless of where the model runs:

- TransformersBackend: loads the base model + your fine-tuned LoRA adapter in
  this process (QLoRA when an NVIDIA GPU is available).
- OllamaBackend: talks to a locally-running Ollama server over HTTP. Great for
  CPU/Apple Silicon, or to serve your merged+GGUF model. See docs/ollama.md.

Pick one via the config 'backend' section or the --backend CLI flag.
"""
import json
import sys
import urllib.error
import urllib.request


class Backend:
    """Common interface: generate(history) -> reply string."""

    name = "base"

    def generate(self, history, max_new_tokens=512, temperature=0.7, top_p=0.9):
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, cfg: dict, use_adapter: bool = True):
        from _common import load_model_and_tokenizer
        self.cfg = cfg
        self.model, self.tokenizer = load_model_and_tokenizer(
            cfg, use_adapter=use_adapter
        )
        self._adapter = use_adapter

    def generate(self, history, max_new_tokens=512, temperature=0.7, top_p=0.9):
        from _common import generate_reply
        return generate_reply(
            self.model, self.tokenizer, history,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        )

    def describe(self) -> str:
        base = self.cfg["model"]["name"]
        return f"transformers · {base}" + (" + adapter" if self._adapter else "")


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, cfg: dict):
        ocfg = cfg.get("backend", {}).get("ollama", {})
        self.host = ocfg.get("host", "http://localhost:11434").rstrip("/")
        self.model = ocfg.get("model", "qwen2.5:3b")
        self._check_server()

    def _check_server(self):
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            sys.exit(
                f"Could not reach Ollama at {self.host} ({e}).\n"
                f"Start it with 'ollama serve' and pull a model "
                f"(e.g. 'ollama pull {self.model}'). See docs/ollama.md."
            )

    def generate(self, history, max_new_tokens=512, temperature=0.7, top_p=0.9):
        payload = {
            "model": self.model,
            "messages": history,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"Ollama error {e.code}: {e.read().decode('utf-8', 'ignore')}")
        return resp.get("message", {}).get("content", "").strip()

    def describe(self) -> str:
        return f"ollama · {self.model} @ {self.host}"


def get_backend(cfg: dict, backend: str = None, use_adapter: bool = True) -> Backend:
    """Factory. 'backend' overrides config['backend']['type'] when provided."""
    kind = backend or cfg.get("backend", {}).get("type", "transformers")
    kind = kind.lower()
    if kind == "ollama":
        return OllamaBackend(cfg)
    if kind == "transformers":
        return TransformersBackend(cfg, use_adapter=use_adapter)
    sys.exit(f"Unknown backend '{kind}'. Use 'transformers' or 'ollama'.")
