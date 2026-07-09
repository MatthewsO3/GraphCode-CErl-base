#!/usr/bin/env python3
"""
setup.py — Bootstrap the code_search environment on a fresh machine.

What it does:
  1. Creates a Python virtual environment (.venv)
  2. Installs all Python dependencies
  3. Clones tree-sitter-erlang and compiles parsers/erlang.so

Usage:
  python setup.py

Requirements:
  - Python 3.9+
  - git
  - gcc or clang  (or the tree-sitter CLI)
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(msg):   print(f"  \033[32m✓\033[0m {msg}")
def warn(msg): print(f"  \033[33m!\033[0m {msg}")
def die(msg):
    print(f"  \033[31m✗\033[0m {msg}", file=sys.stderr)
    sys.exit(1)

def run(cmd, **kwargs):
    """Run a command, raise on failure."""
    subprocess.run(cmd, check=True, **kwargs)

def run_silent(cmd, **kwargs):
    """Run a command, suppress output, raise on failure."""
    subprocess.run(cmd, check=True, capture_output=True, **kwargs)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
VENV_DIR    = SCRIPT_DIR / ".venv"
PARSERS_DIR = SCRIPT_DIR / "parsers"
TS_REPO     = PARSERS_DIR / "tree-sitter-erlang"
SO_PATH     = PARSERS_DIR / "erlang.so"

# Python executable inside the venv
if platform.system() == "Windows":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP    = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP    = VENV_DIR / "bin" / "pip"

# ── Steps ─────────────────────────────────────────────────────────────────────

def step_venv():
    print("\nCreating virtual environment...")
    if VENV_DIR.exists():
        warn(".venv already exists — skipping creation")
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok(f"Virtual environment created at {VENV_DIR.relative_to(SCRIPT_DIR)}")


def step_deps():
    print("\nInstalling Python dependencies...")

    run_silent([str(VENV_PIP), "install", "--upgrade", "pip"])

    required = [
        "torch==2.7",
        "transformers",
        "tree-sitter",
        "tree-sitter-cpp",
        "tree-sitter-python",
        "ranx",
    ]
    run([str(VENV_PIP), "install"] + required)
    ok("Core dependencies installed")

    # Optional — non-fatal if unavailable on this platform
    try:
        run_silent([str(VENV_PIP), "install", "tree-sitter-erlang"])
        ok("tree-sitter-erlang pip package installed")
    except subprocess.CalledProcessError:
        warn("tree-sitter-erlang pip package unavailable — DFG fallback will be used (fine)")


def step_clone():
    print("\nCloning tree-sitter-erlang grammar...")
    PARSERS_DIR.mkdir(exist_ok=True)

    if TS_REPO.exists():
        warn(f"{TS_REPO.relative_to(SCRIPT_DIR)} already exists — skipping clone")
        return

    run_silent([
        "git", "clone", "--depth=1",
        "https://github.com/WhatsApp/tree-sitter-erlang.git",
        str(TS_REPO),
    ])
    ok("Cloned WhatsApp/tree-sitter-erlang")


def step_build_so():
    print("\nCompiling parsers/erlang.so...")

    if SO_PATH.exists():
        warn("parsers/erlang.so already exists — skipping build "
             "(delete it and re-run to rebuild)")
        return

    c_files = list((TS_REPO / "src").glob("*.c"))
    if not c_files:
        die(f"No .c files found in {TS_REPO / 'src'} — cannot compile")

    # Method 1: tree-sitter CLI
    if shutil.which("tree-sitter"):
        try:
            run_silent([
                "tree-sitter", "build",
                "--output", str(SO_PATH),
                str(TS_REPO),
            ], cwd=str(TS_REPO))
            ok("Built erlang.so via tree-sitter CLI")
            return
        except subprocess.CalledProcessError:
            warn("tree-sitter CLI failed — trying compiler directly")

    # Method 2: gcc / clang
    compiler = shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        die("Neither tree-sitter CLI nor gcc/clang found. Install one and re-run.")

    if platform.system() == "Windows":
        die("Windows direct compilation is not supported. Install the tree-sitter CLI instead.")

    cmd = [
        compiler,
        "-shared", "-fPIC", "-O2",
        "-I", str(TS_REPO / "src"),
        "-o", str(SO_PATH),
    ] + [str(f) for f in c_files]

    try:
        run_silent(cmd)
        ok(f"Built erlang.so via {Path(compiler).name}")
    except subprocess.CalledProcessError as e:
        die(f"Compilation failed:\n{e.stderr.decode() if e.stderr else e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.version_info < (3, 9):
        die(f"Python 3.9+ required (you have {platform.python_version()})")

    print()
    print("=" * 44)
    print("  code_search setup")
    print("=" * 44)

    step_venv()
    step_deps()
    step_clone()
    step_build_so()

    activate = (
        r".venv\Scripts\activate" if platform.system() == "Windows"
        else "source .venv/bin/activate"
    )

    print()
    print("=" * 44)
    ok("Setup complete!")
    print()
    print("  Activate the environment with:")
    print(f"    {activate}")
    print()
    print("  Then index a repo:")
    print("    python code_search.py index --repo /path/to/repo \\")
    print("                                --model /path/to/model")
    print("=" * 44)
    print()


if __name__ == "__main__":
    main()