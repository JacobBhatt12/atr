"""Deploy ATR's online connector to Vercel's free Hobby plan.

    npm i -g vercel && vercel login          # once
    python deploy/vercel/deploy.py           # creates or updates the project, prints the link

The first deploy also creates the project's ATR_SECRET, which encrypts personal links. Later
deploys keep it, so links people made keep working. --new-secret replaces it, which breaks
every personal link made so far.

Hobby is free with no card on file: if the connector ever uses more than the monthly free
amount, Vercel pauses it until the month resets. It never bills.
"""
from __future__ import annotations

import argparse
import secrets
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / ".build"  # kept between deploys for its .vercel link; gitignored


def vercel(*args: str, input: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["vercel", *args], cwd=BUILD, input=input, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"vercel {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="atr-mcp")
    p.add_argument("--new-secret", action="store_true", help="replace ATR_SECRET (breaks existing personal links)")
    a = p.parse_args()

    BUILD.mkdir(exist_ok=True)
    shutil.rmtree(BUILD / "atr", ignore_errors=True)
    shutil.copytree(ROOT / "atr", BUILD / "atr", ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    for f in ("app.py", "requirements.txt"):
        shutil.copy(HERE / f, BUILD / f)
    (BUILD / ".python-version").write_text("3.12\n")

    if not (BUILD / ".vercel" / "project.json").exists():
        vercel("link", "--yes", "--project", a.project)
    has_secret = "ATR_SECRET" in vercel("env", "ls", "production", check=False)
    if a.new_secret and has_secret:
        vercel("env", "rm", "ATR_SECRET", "production", "--yes")
    if a.new_secret or not has_secret:
        vercel("env", "add", "ATR_SECRET", "production", "--sensitive", input=secrets.token_urlsafe(48) + "\n")

    out = vercel("deploy", "--prod", "--yes")
    print(out.strip().splitlines()[-1] if out.strip() else out)
    print(f"Connector link: https://{a.project}.vercel.app/mcp  (home page: https://{a.project}.vercel.app)")


if __name__ == "__main__":
    main()
