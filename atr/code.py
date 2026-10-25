"""Coding tools: the parts of a coding session that burn the most tokens, made small.

  repo_map     the project's layout and the main functions/classes in each file
  search_code  regex search across the project, grouped by file and capped
  run          a shell command (tests, builds, linters) with its output compressed

Every result goes through the same compression and budget as other tool results; anything
cut is stashed, so atr_query can page through the rest.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from atr import tokens

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".next", ".nuxt",
             "target", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vscode", "coverage",
             ".gradle", ".cache", "vendor", "Pods", ".turbo", ".parcel-cache", ".svelte-kit", "out"}
CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".c",
            ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".scala", ".lua", ".sh", ".vue", ".svelte", ".dart"}
TEXT_EXT = CODE_EXT | {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".html", ".css", ".scss",
                       ".sql", ".graphql", ".proto", ".env.example", ".xml", ".gradle", ".dockerfile", ""}
# top-level definitions per language (no indentation = top level)
SYMBOL = [
    re.compile(r"^(?:async\s+def|def|class)\s+(\w+)"),                                        # Python
    re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s+(\w+)"),  # JS/TS
    re.compile(r"^(?:export\s+)(?:const|let)\s+(\w+)"),                                       # exported consts
    re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)"), re.compile(r"^type\s+(\w+)"),              # Go
    re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|mod)\s+(\w+)"),  # Rust
    re.compile(r"^(?:public\s+|internal\s+|abstract\s+|final\s+|sealed\s+|data\s+|open\s+)*(?:class|interface|object|enum|struct)\s+(\w+)"),
]


SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".atr", ".thrift-ai", ".kube", ".docker", ".azure", ".config/gcloud", ".password-store"}
SECRET_FILE = re.compile(r"^(\.env(\..*)?|.+\.env|\.netrc|\.git-credentials|\.npmrc|\.pypirc|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?"
                         r"|.*\.(pem|key|p12|pfx|keystore|jks)|credentials(\.json)?|secrets?\.(json|ya?ml|toml)"
                         r"|http_token|.*\.sqlite3?)$", re.I)


SECRET_WORDS = re.compile(r"secret|access[-_ ]?keys?|api[-_ ]?keys?|apikey|passw(or)?d|credential|private[-_ ]?key"
                          r"|auth[-_ ]?token|(tokens?|keys?)\.(txt|json|ini|ya?ml|csv|toml|cfg)$|keychain|vault", re.I)


def is_secret(path: Path) -> bool:
    """Never shown by ATR: keys, tokens, .env files, cloud credentials (example files are fine)."""
    parts = path.parts
    if any(p in SECRET_DIRS for p in parts) or ".config/gcloud" in path.as_posix():
        return True
    # name rules are for files; a folder called "secrets-demo" is just a folder. Only ask the disk about
    # full paths: a relative path would be looked up from wherever ATR was started, not the project.
    if path.is_absolute() and path.is_dir():
        return False
    name = path.name
    if name.endswith((".example", ".sample", ".template")):
        return False
    if SECRET_FILE.match(name):
        return True
    # data/config files whose name says what they hold (code files like tokens.py are fine)
    return path.suffix.lower() not in CODE_EXT and bool(SECRET_WORDS.search(name))


def _ignored(rel: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".venv") for part in rel.parts[:-1])


def list_files(root: Path, limit: int = 20000) -> list[Path]:
    """Project files, relative to root. Uses git (respects .gitignore) when root is in a repo."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            files: list[Path] = []
            for f in out.stdout.splitlines():
                p = Path(f)
                if not f or is_secret(root / p):
                    continue
                if f.endswith("/") and (root / p).is_dir():
                    if _ignored(p / "x"):
                        continue
                    # git shows a nested repo as one folder entry; list that repo's own files
                    files += [p / sub for sub in list_files(root / p, limit - len(files))]
                elif not _ignored(p) and (root / p).is_file():
                    files.append(p)
                if len(files) >= limit:
                    break
            return files[:limit]
    except (OSError, subprocess.TimeoutExpired):
        pass
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv") and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root)
        for f in sorted(filenames):
            if is_secret(rel_dir / f):
                continue
            files.append(rel_dir / f)
            if len(files) >= limit:
                return files
    return files


def _is_text(p: Path) -> bool:
    if (p.suffix.lower() in TEXT_EXT or p.suffix.lower() in (".example", ".sample", ".template")
            or p.name in ("Dockerfile", "Makefile", "Procfile", ".gitignore")):
        try:
            with open(p, "rb") as f:
                return b"\0" not in f.read(2048)
        except OSError:
            return False
    return False


def symbols(path: Path, limit: int = 12) -> tuple[int, list[str]]:
    """(line count, top-level names) for a code file."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return 0, []
    names: list[str] = []
    for ln in text.splitlines():
        if ln[:1] in (" ", "\t", "#", "/", "*", "@") or not ln.strip():
            continue
        for rx in SYMBOL:
            m = rx.match(ln)
            if m and m.group(1) not in names:
                names.append(m.group(1))
                break
    more = len(names) - limit
    return len(text.splitlines()), names[:limit] + ([f"+{more}"] if more > 0 else [])


def repo_map(root: Path, budget: int) -> str:
    files = list_files(root)
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_dir[f.parent].append(f)

    def render(with_symbols: bool, max_depth: int | None, symbol_depth: int = 99) -> str:
        out = [f"{root.name}/ ({len(files)} files)"]
        for d in sorted(by_dir):
            depth = len(d.parts)
            if max_depth is not None and depth > max_depth:
                continue
            fs = by_dir[d]
            if max_depth is not None and depth == max_depth:
                nested = sum(len(v) for k, v in by_dir.items() if k == d or d in k.parents)
                out.append(f"{'  ' * depth}{d.as_posix()}/ ({nested} files)")
                continue
            if d.parts:
                out.append(f"{'  ' * (depth - 1)}{d.as_posix()}/")
            pad = "  " * depth
            plain = []
            for f in sorted(fs):
                if with_symbols and depth <= symbol_depth and f.suffix.lower() in CODE_EXT:
                    n, names = symbols(root / f)
                    out.append(f"{pad}{f.name} {n}L" + (f": {', '.join(names)}" if names else ""))
                else:
                    plain.append(f.name)
            if plain:
                shown = plain[:15]
                out.append(pad + ", ".join(shown) + (f", +{len(plain) - 15} more" if len(plain) > 15 else ""))
        return "\n".join(out)

    # shrink step by step: functions/classes only near the top, then names only, then fewer levels
    steps = [(True, None, 99), (True, None, 1), (True, None, 0), (False, None, 0), (False, 3, 0), (False, 2, 0), (False, 1, 0)]
    for with_symbols, depth, sym_depth in steps:
        text = render(with_symbols, depth, sym_depth)
        if tokens.count(text) <= budget:
            if (with_symbols, sym_depth) != (True, 99):
                text += "\n[atr: map shortened to fit; pass path= a subfolder to see its functions and classes]"
            return text
    return text + "\n[atr: still large; pass path= a subfolder]"


def search_code(root: Path, pattern: str, glob: str | None, budget: int, per_file: int = 6, context: int = 0) -> str:
    """file -> [(line, text)] for a regex, via ripgrep when installed. context=N adds N lines around each match."""
    hits: dict[str, list[tuple[int, str]]] = defaultdict(list)
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "300", "-i", "-e", pattern,
               "-g", "!.env*", "-g", "!*.pem", "-g", "!*.key", "-g", "!id_*"]
        if glob:
            cmd += ["-g", glob]
        out = subprocess.run(cmd + ["."], cwd=root, capture_output=True, text=True, timeout=60)
        for ln in out.stdout.splitlines():
            parts = ln.split(":", 2)
            if len(parts) == 3 and parts[1].isdigit() and not is_secret(Path(parts[0])):
                hits[parts[0].removeprefix("./")].append((int(parts[1]), parts[2]))
    else:
        rx = re.compile(pattern, re.I)
        for f in list_files(root):
            if glob and not (f.match(glob) or Path(f.name).match(glob)):
                continue
            p = root / f
            if not _is_text(p):
                continue
            try:
                for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(ln):
                        hits[f.as_posix()].append((i, ln))
            except OSError:
                continue
    total = sum(len(v) for v in hits.values())
    if not total:
        return f"no matches for {pattern!r}" + (f" in {glob}" if glob else "")
    lines = [f"{total} matches in {len(hits)} files"]
    shown_files = 0
    for f in sorted(hits, key=lambda k: (-len(hits[k]), k)):
        block = [f] + (_with_context(root / f, [n for n, _ in hits[f][:per_file]], context) if context > 0
                       else [f"  {n}: {t.strip()[:160]}" for n, t in hits[f][:per_file]])
        if len(hits[f]) > per_file:
            block.append(f"  … +{len(hits[f]) - per_file} more in this file")
        if tokens.count("\n".join(lines + block)) > budget:
            break
        lines += block
        shown_files += 1
    if shown_files < len(hits):
        lines.append(f"… +{len(hits) - shown_files} more files; narrow it with glob= or a more specific pattern")
    return "\n".join(lines)


def _with_context(path: Path, line_numbers: list[int], context: int) -> list[str]:
    """Matching lines plus `context` lines around each, indentation kept, "--" between separate chunks."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return [f"  {n}: (unreadable)" for n in line_numbers]
    keep: list[int] = []
    for n in line_numbers:
        for j in range(max(1, n - context), min(len(lines), n + context) + 1):
            if not keep or j > keep[-1]:
                if keep and j > keep[-1] + 1:
                    keep.append(-1)
                keep.append(j)
    return ["  --" if j < 0 else f"  {j}{':' if j in line_numbers else '-'} {lines[j - 1].rstrip()[:200]}" for j in keep]


async def run(command: str, cwd: Path, timeout: float) -> tuple[int | None, str, float]:
    """(exit code or None on timeout, combined output, seconds)."""
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(command, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
                                                 start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        code = proc.returncode
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = await proc.communicate()
        code = None
    return code, out.decode(errors="replace"), time.monotonic() - t0
