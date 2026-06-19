#!/usr/bin/env python
"""Turn your GitHub repositories into a coding training dataset for MyGPT.

It (1) lists your repos via the GitHub API, (2) shallow-clones each one, and
(3) converts your source files into chat-format training examples that teach the
model your coding style and patterns.

Two example types are produced:
  - completion: given the first part of one of your files, continue it. This is
    the strongest signal for learning your style. (all languages)
  - py-docstring: given a function's docstring, write the function. High-quality
    instruction pairs. (Python only, via the stdlib `ast` module)

Usage:
  # All your repos (needs a token with 'repo' scope to include private ones):
  export GITHUB_TOKEN=ghp_xxx
  python scripts/github_dataset.py --user hassanmujtaba22 --out data/code_train.jsonl

  # Limit / filter while experimenting:
  python scripts/github_dataset.py --user hassanmujtaba22 --max-repos 5 \
      --include-private false --out data/code_train.jsonl

Nothing leaves your machine: cloning and processing are entirely local. Review
the output before training and avoid committing private code / secrets.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# What to include / exclude
# ---------------------------------------------------------------------------

# extension -> language label used in prompts
LANG_BY_EXT = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript (React)",
    ".js": "JavaScript", ".jsx": "JavaScript (React)", ".mjs": "JavaScript",
    ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".cs": "C#",
    ".css": "CSS", ".scss": "SCSS", ".html": "HTML", ".vue": "Vue", ".svelte": "Svelte",
    ".sql": "SQL", ".sh": "Shell", ".yml": "YAML", ".yaml": "YAML",
    ".kt": "Kotlin", ".swift": "Swift", ".dart": "Dart",
}

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "out", ".next", ".nuxt", "vendor",
    "venv", ".venv", "__pycache__", ".cache", "coverage", ".idea", ".vscode",
    "migrations", "public", "assets", ".turbo", "target", "bin", "obj",
}
# Filenames / suffixes we never want as training data.
SKIP_NAME_HINTS = (".min.", ".lock", ".map", ".d.ts", ".test.", ".spec.")
SKIP_EXACT = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "composer.lock", "Gemfile.lock",
}
# Files that may hold secrets — never include.
SECRET_HINTS = (".env", "secret", "credential", ".pem", ".key", "id_rsa")

SYSTEM_PROMPT = (
    "You are MyGPT, an expert pair programmer fine-tuned on the user's own code. "
    "You write clean, idiomatic code that matches their style and conventions."
)


# ---------------------------------------------------------------------------
# GitHub API: list repositories
# ---------------------------------------------------------------------------

def list_repos(user: str, token: str, include_private: bool, include_forks: bool):
    """Return a list of {full_name, clone_url, default_branch, language}."""
    repos, page = [], 1
    # Authenticated /user/repos sees private repos; otherwise public only.
    base = "https://api.github.com/user/repos" if token else \
           f"https://api.github.com/users/{user}/repos"
    while True:
        url = f"{base}?per_page=100&page={page}&sort=updated"
        if not token:
            url += "&type=owner"
        req = urllib.request.Request(url, headers=_gh_headers(token))
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                batch = json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"GitHub API error {e.code}: {e.read().decode('utf-8','ignore')}")
        if not batch:
            break
        for repo in batch:
            if repo.get("owner", {}).get("login", "").lower() != user.lower():
                continue  # only repos the user owns
            if repo.get("fork") and not include_forks:
                continue
            if repo.get("private") and not include_private:
                continue
            repos.append({
                "full_name": repo["full_name"],
                "clone_url": repo["clone_url"],
                "default_branch": repo.get("default_branch", "main"),
                "language": repo.get("language"),
            })
        page += 1
    return repos


def _gh_headers(token: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "MyGPT"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ---------------------------------------------------------------------------
# Clone + walk
# ---------------------------------------------------------------------------

def clone_repo(clone_url: str, token: str, dest: Path) -> bool:
    """Shallow-clone a repo. Returns True on success."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    url = clone_url
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://x-access-token:{token}@")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True, capture_output=True, timeout=300,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    clone failed: {getattr(e, 'stderr', b'') and e.stderr.decode('utf-8','ignore')[:200] or e}")
        return False


def iter_source_files(root: Path, max_bytes: int):
    """Yield (path, language, text) for usable source files under root."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        name = path.name.lower()
        if name in SKIP_EXACT or any(h in name for h in SKIP_NAME_HINTS):
            continue
        if any(h in name for h in SECRET_HINTS):
            continue
        ext = path.suffix.lower()
        lang = LANG_BY_EXT.get(ext)
        if not lang:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip() or "\x00" in text[:1000]:
            continue
        yield path, lang, text


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------

def make_completion_example(rel_path, repo, lang, text, min_lines):
    """Teach the model to continue your code: prefix -> the rest of the file."""
    lines = text.splitlines()
    if len(lines) < min_lines:
        return None
    split = max(min_lines // 2, int(len(lines) * 0.4))
    prefix = "\n".join(lines[:split])
    suffix = "\n".join(lines[split:])
    if not suffix.strip():
        return None
    user = (
        f"Continue this {lang} file `{rel_path}` from my `{repo}` project. "
        f"Complete it in my style.\n\n```\n{prefix}\n```"
    )
    return _chat(user, f"```\n{suffix}\n```")


def make_python_docstring_examples(rel_path, repo, text):
    """Extract (docstring -> function) pairs from a Python file via AST."""
    import ast
    examples = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return examples
    src_lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or len(doc) < 25:
            continue
        end = getattr(node, "end_lineno", None)
        if not end:
            continue
        func_src = "\n".join(src_lines[node.lineno - 1:end])
        if func_src.count("\n") < 3:   # skip trivial one-liners
            continue
        user = (
            f"Write a Python function named `{node.name}` for my `{repo}` project "
            f"that does the following:\n\n{doc.strip()}"
        )
        examples.append(_chat(user, f"```python\n{func_src}\n```"))
    return examples


def _chat(user_content, assistant_content):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _bool(v):
    return str(v).lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser(description="Build a coding dataset from your GitHub repos.")
    ap.add_argument("--user", required=True, help="Your GitHub username.")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                    help="GitHub token (or set GITHUB_TOKEN). Needed for private repos.")
    ap.add_argument("--out", default="data/code_train.jsonl")
    ap.add_argument("--workdir", default=".cache/repos", help="Where repos are cloned.")
    ap.add_argument("--include-private", type=_bool, default=True)
    ap.add_argument("--include-forks", type=_bool, default=False)
    ap.add_argument("--max-repos", type=int, default=0, help="0 = all repos.")
    ap.add_argument("--max-file-kb", type=int, default=120,
                    help="Skip files larger than this many KB.")
    ap.add_argument("--min-lines", type=int, default=12,
                    help="Minimum lines for a completion example.")
    ap.add_argument("--max-per-repo", type=int, default=200,
                    help="Cap examples taken from any single repo (balance).")
    ap.add_argument("--modes", default="completion,py-docstring",
                    help="Comma list: completion, py-docstring.")
    ap.add_argument("--keep-clones", action="store_true",
                    help="Don't delete cloned repos after processing.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if shutil.which("git") is None:
        sys.exit("git is required and was not found on PATH.")
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    random.seed(args.seed)

    if not args.token:
        print("No token provided — only PUBLIC repos will be included.\n"
              "Set GITHUB_TOKEN (scope 'repo') to include private repos.\n")

    print(f"Listing repos for {args.user} ...")
    repos = list_repos(args.user, args.token, args.include_private, args.include_forks)
    if args.max_repos:
        repos = repos[:args.max_repos]
    print(f"  {len(repos)} repo(s) to process.\n")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = args.max_file_kb * 1024
    all_examples, stats = [], {"repos_ok": 0, "files": 0, "by_lang": {}}

    for i, repo in enumerate(repos, start=1):
        name = repo["full_name"]
        print(f"[{i}/{len(repos)}] {name}")
        dest = workdir / repo["full_name"].replace("/", "__")
        if not clone_repo(repo["clone_url"], args.token, dest):
            continue
        stats["repos_ok"] += 1

        repo_examples = []
        for path, lang, text in iter_source_files(dest, max_bytes):
            rel = path.relative_to(dest)
            stats["files"] += 1
            stats["by_lang"][lang] = stats["by_lang"].get(lang, 0) + 1
            if "completion" in modes:
                ex = make_completion_example(rel, repo["full_name"].split("/")[-1],
                                             lang, text, args.min_lines)
                if ex:
                    repo_examples.append(ex)
            if "py-docstring" in modes and path.suffix.lower() == ".py":
                repo_examples.extend(make_python_docstring_examples(
                    rel, repo["full_name"].split("/")[-1], text))

        random.shuffle(repo_examples)
        repo_examples = repo_examples[:args.max_per_repo]
        all_examples.extend(repo_examples)
        print(f"    +{len(repo_examples)} examples")

        if not args.keep_clones:
            shutil.rmtree(dest, ignore_errors=True)

    random.shuffle(all_examples)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n" + "=" * 56)
    print(f"Wrote {len(all_examples)} examples to {out_path}")
    print(f"Repos processed: {stats['repos_ok']}/{len(repos)} | "
          f"source files: {stats['files']}")
    if stats["by_lang"]:
        top = sorted(stats["by_lang"].items(), key=lambda kv: -kv[1])[:8]
        print("Top languages: " + ", ".join(f"{l} ({n})" for l, n in top))
    print("\nNext:")
    print("  python scripts/prepare_data.py --config configs/coding.yaml")
    print("  python scripts/train.py        --config configs/coding.yaml")


if __name__ == "__main__":
    main()
