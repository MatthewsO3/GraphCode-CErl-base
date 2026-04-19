"""run.py — end-to-end pipeline for GraphCode-CErl-base.

Runs the four stages in order:
    1. setup        — installs dependencies and checks the environment
    2. preprocess   — downloads and tokenises the training corpus
    3. train        — fine-tunes the model
    4. evaluate     — reports MLM accuracy and perplexity

Every value is read from ``config.json`` by default.  Any key can be
overridden on the command line, e.g.::

    python run.py --batch_size 16 --epochs 3
    python run.py --skip setup preprocess   # jump straight to training
    python run.py --only evaluate           # run a single stage

Stage names accepted by ``--skip`` / ``--only``:
    setup, preprocess, train, evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.json")

# Flat map of every CLI knob → the config section it lives in.
# Format: "arg_name": ("section", "key_in_section")
# Preprocessing args are stored under a top-level "preprocess" section that
# run.py manages itself (they are not present in the original config.json,
# which only has "train" and "evaluate").
_ARG_SECTION_MAP: dict[str, tuple[str, str]] = {
    # --- preprocess ---
    "lang": ("preprocess", "lang"),
    "max_samples": ("preprocess", "max_samples"),
    # --- train ---
    "data_file": ("train", "data_file"),
    "output_dir": ("train", "output_dir"),
    "checkpoint_path": ("train", "checkpoint_path"),
    "batch_size": ("train", "batch_size"),
    "epochs": ("train", "epochs"),
    "learning_rate": ("train", "learning_rate"),
    "max_length": ("train", "max_length"),
    "warmup_steps": ("train", "warmup_steps"),
    "mlm_probability": ("train", "mlm_probability"),
    "validation_split": ("train", "validation_split"),
    "weight_decay": ("train", "weight_decay"),
    "early_stopping_patience": ("train", "early_stopping_patience"),
    # --- evaluate ---
    "model": ("evaluate", "model"),
    "mask_ratio": ("evaluate", "mask_ratio"),
    "top_k": ("evaluate", "top_k"),
    "max_examples": ("evaluate", "max_examples"),
    "langs": ("evaluate", "langs"),
    "data_files": ("evaluate", "data_files"),
}


def load_config() -> dict[str, Any]:
    """Load and return the full ``config.json`` as a dict.

    Returns an empty dict when the file does not exist so that the pipeline
    can still run when only CLI flags are supplied.
    """
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as fh:
            return json.load(fh)
    print(f"[run.py] Warning: {CONFIG_PATH} not found — using CLI/defaults only.")
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    """Write *cfg* back to ``config.json`` (pretty-printed)."""
    with CONFIG_PATH.open("w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[run.py] Updated {CONFIG_PATH} with CLI overrides.")


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI arguments that were explicitly supplied into *cfg*.

    Only arguments whose value is not ``None`` (i.e. actually provided on the
    command line) are written back, so config.json defaults are preserved for
    anything the user did not specify.
    """
    for arg_name, (section, key) in _ARG_SECTION_MAP.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        cfg.setdefault(section, {})
        cfg[section][key] = value
    return cfg


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline control
    p.add_argument(
        "--skip",
        nargs="+",
        metavar="STAGE",
        default=[],
        choices=["setup", "preprocess", "train", "evaluate"],
        help="Stages to skip (space-separated). E.g. --skip setup preprocess",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="STAGE",
        default=[],
        choices=["setup", "preprocess", "train", "evaluate"],
        help="Run only these stages and skip all others.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.json",
        metavar="PATH",
        help="Path to config.json (default: config.json)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the subprocess commands that would be run without executing them.",
    )

    # ---- preprocess --------------------------------------------------------
    g = p.add_argument_group("preprocess")
    g.add_argument(
        "--lang",
        type=str,
        default=None,
        metavar="LANG",
        help="Language to preprocess: cpp | python | java | javascript",
    )
    g.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of samples to write during preprocessing.",
    )

    # ---- train -------------------------------------------------------------
    g = p.add_argument_group("train")
    g.add_argument("--data_file", type=str, default=None)
    g.add_argument("--output_dir", type=str, default=None)
    g.add_argument("--checkpoint_path", type=str, default=None)
    g.add_argument("--batch_size", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--learning_rate", type=float, default=None)
    g.add_argument("--max_length", type=int, default=None)
    g.add_argument("--warmup_steps", type=int, default=None)
    g.add_argument("--mlm_probability", type=float, default=None)
    g.add_argument("--validation_split", type=float, default=None)
    g.add_argument("--weight_decay", type=float, default=None)
    g.add_argument("--early_stopping_patience", type=int, default=None)

    # ---- evaluate ----------------------------------------------------------
    g = p.add_argument_group("evaluate")
    g.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID or local checkpoint path.",
    )
    g.add_argument("--mask_ratio", type=float, default=None)
    g.add_argument("--top_k", type=int, default=None)
    g.add_argument("--max_examples", type=int, default=None)
    g.add_argument(
        "--langs",
        type=str,
        nargs="+",
        default=None,
        metavar="LANG",
        help="Languages to evaluate (space-separated).",
    )
    g.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=None,
        metavar="PATH",
        help="JSONL data files for evaluation (parallel to --langs).",
    )

    return p


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, dry_run: bool = False, stage: str = "") -> None:
    """Execute *cmd* as a subprocess, or print it when *dry_run* is True."""
    label = f"[{stage}] " if stage else ""
    pretty = " ".join(cmd)
    print(f"\n{label}$ {pretty}\n{'─' * 72}")
    if dry_run:
        print("  (dry-run — not executing)")
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"\n[run.py] ✗ Stage '{stage}' exited with code {result.returncode}. "
            "Aborting pipeline."
        )
        sys.exit(result.returncode)


def run_setup(cfg: dict[str, Any], *, dry_run: bool) -> None:
    _run([sys.executable, "setup.py"], dry_run=dry_run, stage="setup")


def run_preprocess(cfg: dict[str, Any], *, dry_run: bool) -> None:
    # preprocessing.py now reads config.json itself via --config, so we only
    # need to forward the config path.  Any CLI overrides have already been
    # written back into config.json by apply_overrides().
    cmd = [sys.executable, "preprocessing.py", "--config", str(CONFIG_PATH)]
    _run(cmd, dry_run=dry_run, stage="preprocess")


def run_train(cfg: dict[str, Any], *, dry_run: bool) -> None:
    # train.py reads config.json itself via load_config_and_set_defaults, so
    # we only need to forward the config path and any extra CLI overrides that
    # have already been written back into config.json by apply_overrides.
    # Passing --config keeps things explicit.
    train_cfg = cfg.get("train", {})

    cmd = [sys.executable, "train.py"]

    # Forward every known train argument so CLI overrides reach the script
    # even if the user is running without a config.json present.
    _opt_str = {
        "data_file": "--data_file",
        "output_dir": "--output_dir",
        "checkpoint_path": "--checkpoint_path",
    }
    _opt_int = {
        "batch_size": "--batch_size",
        "epochs": "--epochs",
        "max_length": "--max_length",
        "warmup_steps": "--warmup_steps",
        "early_stopping_patience": "--early_stopping_patience",
    }
    _opt_float = {
        "learning_rate": "--learning_rate",
        "mlm_probability": "--mlm_probability",
        "validation_split": "--validation_split",
        "weight_decay": "--weight_decay",
    }

    for key, flag in {**_opt_str, **_opt_int, **_opt_float}.items():
        val = train_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]

    _run(cmd, dry_run=dry_run, stage="train")


def run_evaluate(cfg: dict[str, Any], *, dry_run: bool) -> None:
    # evaluate.py reads everything from config.json via --config.
    cmd = [sys.executable, "evaluate.py", "--config", str(CONFIG_PATH)]
    _run(cmd, dry_run=dry_run, stage="evaluate")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

STAGES: list[tuple[str, Any]] = [
    ("setup", run_setup),
    ("preprocess", run_preprocess),
    ("train", run_train),
    ("evaluate", run_evaluate),
]


def resolve_stages(
    skip: list[str],
    only: list[str],
) -> list[tuple[str, Any]]:
    """Return the ordered list of (name, runner) pairs to execute."""
    if only:
        return [(name, fn) for name, fn in STAGES if name in only]
    return [(name, fn) for name, fn in STAGES if name not in skip]


def print_plan(stages: list[tuple[str, Any]]) -> None:
    all_names = [name for name, _ in STAGES]
    run_names = {name for name, _ in stages}
    parts = []
    for name in all_names:
        parts.append(name if name in run_names else f"({name})")
    print(f"[run.py] Pipeline: {' → '.join(parts)}")
    print(f"         Stages to run: {', '.join(run_names) or '(none)'}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Allow --config to point elsewhere
    global CONFIG_PATH
    CONFIG_PATH = Path(args.config)

    cfg = load_config()
    cfg = apply_overrides(cfg, args)

    # Persist overrides so that sub-scripts (train.py, evaluate.py) that read
    # config.json directly also see the updated values.
    if CONFIG_PATH.exists() or any(
        getattr(args, k, None) is not None for k in _ARG_SECTION_MAP
    ):
        save_config(cfg)

    stages = resolve_stages(skip=args.skip, only=args.only)
    print_plan(stages)

    for name, runner in stages:
        print(f"\n{'=' * 72}")
        print(f"  STAGE: {name.upper()}")
        print(f"{'=' * 72}")
        runner(cfg, dry_run=args.dry_run)
        print(f"\n[run.py] ✓ Stage '{name}' completed.")

    print(f"\n{'=' * 72}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
