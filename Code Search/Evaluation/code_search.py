#!/usr/bin/env python3
"""
code_search.py — Unified code indexer and semantic search tool.

Modes
-----
index   Parse a local Erlang and/or C++ repository, extract functions/snippets,
        and write a JSONL corpus + an embedding index to disk.

search  Load a fine-tuned GraphCodeBERT model and the pre-built index, then
        answer free-text queries with the top-N matching functions.

Usage examples
--------------
# Multilingual repo: auto-discover Erlang + C++ + Python under one root
python code_search.py index --repo "/path/to/mixed_repo" --model "/path/to/best_model" --output corpus.jsonl --index corpus_index.pt

# Language-specific flags (combinable; --repo is additive with the others)
python code_search.py index --erlang "..." --cpp "..." --python "..." --model "MatthewsO3/GraphCode-CErl-codesearch" --output three_lang.jsonl --index three_lang_index.pt

# Python-only
python code_search.py index --python "..." --model "MatthewsO3/GraphCode-CErl-codesearch" --output py.jsonl --index py_index.pt

# Search against a pre-built corpus
python code_search.py search --model "MatthewsO3/GraphCode-CErl-codesearch" --jsonl corpus.jsonl --index corpus_index.pt --top 5

# lenient: any graded function entry counts as valid gold
python code_search.py eval --model "MatthewsO3/GraphCode-CErl-codesearch" --jsonl corpus.jsonl --index corpus_index.pt --queries example.jsonl --min-relevance 1 --output eval_lenient.jsonl

# strict: only relevance=5 function entries count
python code_search.py eval --model "MatthewsO3/GraphCode-CErl-codesearch" --jsonl corpus.jsonl --index corpus_index.pt --queries example.jsonl --min-relevance 5 --output eval_strict.jsonl
"""


from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional heavy imports – deferred so that `index` mode does not require
# PyTorch and `search` mode does not require tree-sitter grammars for
# languages the user did not install.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 – Language-agnostic helpers
# ═══════════════════════════════════════════════════════════════════════════

def _simple_tokenize(code: str) -> List[str]:
    """Very light tokeniser used as a fallback when tree-sitter is unavailable."""
    return re.findall(r"[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]", code)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 – DFG extraction (shared with evaluate script)
# ═══════════════════════════════════════════════════════════════════════════

def _build_dfg(code: str, lang: str, parser, ts_langs: dict) -> List[Tuple]:
    """
    Extract a lightweight dataflow graph from *code* for the given *lang*.

    Returns a list of 5-tuples compatible with the GraphCodeBERT format:
        (var_name, use_token_pos, "comesFrom", [var_name], [def_token_pos])

    Falls back to an empty list when tree-sitter is unavailable.
    """
    if parser is None or lang not in ts_langs:
        return []

    import inspect
    # tree-sitter >= 0.22: language is bound at construction time
    if inspect.isclass(parser):
        parser = parser(ts_langs[lang])
    else:
        parser = type(parser)(ts_langs[lang])
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    tokens: List[Any] = []
    node_map: Dict[int, int] = {}

    def collect_tokens(node: Any) -> None:
        if node.type in ("identifier", "variable"):
            if id(node) not in node_map:
                node_map[id(node)] = len(tokens)
                tokens.append(node)
        for child in node.children:
            collect_tokens(child)

    collect_tokens(root)

    defs: Dict[str, List[int]] = defaultdict(list)
    uses: Dict[str, List[int]] = defaultdict(list)

    def _is_def(node: Any) -> bool:
        p = node.parent
        if not p:
            return False
        if lang == "cpp":
            return p.type in ("declaration", "parameter_declaration") or (
                p.type == "assignment_expression"
                and node == p.child_by_field_name("left")
            )
        if lang == "erlang":
            return p.type == "variable" and p.parent is not None and p.parent.type in (
                "match_expression", "clause"
            )
        return False

    def collect_vars(node: Any) -> None:
        if node.type in ("identifier", "variable"):
            name = code_bytes[node.start_byte : node.end_byte].decode("utf-8", "ignore")
            pos = node_map.get(id(node), -1)
            if pos != -1:
                (defs if _is_def(node) else uses)[name].append(pos)
        for child in node.children:
            collect_vars(child)

    collect_vars(root)

    edges: List[Tuple] = []
    for name, use_positions in uses.items():
        def_positions = sorted(defs.get(name, []))
        for use_pos in use_positions:
            preds = [d for d in def_positions if d < use_pos]
            if preds:
                edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))
    return edges


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 – C++ extractor
# ═══════════════════════════════════════════════════════════════════════════

CPP_EXTENSIONS = {".cpp", ".cxx", ".cc", ".c", ".hpp", ".hxx", ".h"}
SKIP_DIRS = {".git", ".svn", "build", "_build", "CMakeFiles", "node_modules",
             "third_party", "vendor", "deps", "ebin",
             # Python-specific
             ".venv", "venv", "env", ".env", "__pycache__", ".mypy_cache",
             ".pytest_cache", "dist", ".tox", "site-packages"}


class CppExtractor:
    """
    Extracts free functions and member functions from C++ source files.

    Uses tree-sitter when available; otherwise falls back to a regex-based
    heuristic that captures the first BLOCK_LINES lines of each
    `return_type name(…) {` pattern.
    """

    # No hard line cap – brace-depth scan determines the real end of every function.

    def __init__(self, ts_parser=None, ts_langs: dict = None):
        self._ParserClass = ts_parser  # tree-sitter Parser class (>=0.22: Parser(lang))
        self._ts_langs = ts_langs or {}
        self._use_ts = ts_parser is not None and "cpp" in self._ts_langs

    # ------------------------------------------------------------------
    def extract_from_repo(self, repo_path: str, repo_name: str) -> List[Dict]:
        """Walk *repo_path* and extract functions from every C++ file."""
        repo_path = Path(repo_path)
        records: List[Dict] = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in CPP_EXTENSIONS:
                    continue
                try:
                    records.extend(
                        self._extract_from_file(fpath, repo_name, repo_path)
                    )
                except Exception as exc:
                    log.warning("Skipping %s: %s", fpath, exc)
        log.info("C++ extractor: %d functions from %s", len(records), repo_name)
        return records

    # ------------------------------------------------------------------
    def _extract_from_file(
        self, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if self._use_ts:
            return self._extract_ts(content, fpath, repo_name, repo_root)
        result = self._extract_regex(content, fpath, repo_name, repo_root)
        log.info("  %s → %d functions (regex)", fpath.name, len(result))
        return result

    # ── tree-sitter path ──────────────────────────────────────────────
    def _extract_ts(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        parser = self._ParserClass(self._ts_langs["cpp"])
        tree = parser.parse(content.encode("utf-8"))
        lines = content.splitlines()
        records: List[Dict] = []

        def visit(node):
            # Only capture top-level function_definition nodes to avoid
            # double-counting nested lambdas / member function bodies.
            if node.type == "function_definition":
                try:
                    start = node.start_point[0]
                    end = node.end_point[0]
                    code = "\n".join(lines[start : end + 1])
                    name = self._ts_func_name(node, content.encode("utf-8"))
                    dfg = _build_dfg(code, "cpp", self._ParserClass, self._ts_langs)
                    records.append(
                        self._make_record(
                            code=code,
                            name=name,
                            lang="cpp",
                            fpath=fpath,
                            repo_name=repo_name,
                            repo_root=repo_root,
                            line_start=start + 1,
                            line_end=end + 1,
                            dfg=dfg,
                        )
                    )
                except Exception as exc:
                    log.debug("C++ ts extraction error in %s node at line %d: %s", fpath, node.start_point[0], exc)
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        log.info("  %s → %d functions (tree-sitter)", fpath.name, len(records))
        return records

    def _ts_func_name(self, node, code_bytes: bytes) -> str:
        """Best-effort function name from a tree-sitter node."""
        # Look for a child whose type is 'function_declarator' or 'identifier'
        for child in node.children:
            if child.type == "function_declarator":
                for gc in child.children:
                    if gc.type in ("identifier", "qualified_identifier", "destructor_name"):
                        return code_bytes[gc.start_byte : gc.end_byte].decode("utf-8", "ignore")
            if child.type == "identifier":
                return code_bytes[child.start_byte : child.end_byte].decode("utf-8", "ignore")
        return "<unknown>"

    # ── regex fallback path ───────────────────────────────────────────
    _FUNC_RE = re.compile(
        r"""
        ^                            # start of line
        (?:[\w:*&<>\s]+?\s+)?        # optional return type
        ([\w:~]+)                    # function name  (group 1)
        \s*\(                        # opening paren
        [^)]*\)                      # parameter list
        (?:\s*const)?(?:\s*noexcept)?
        \s*\{                        # opening brace
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def _extract_regex(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        lines = content.splitlines()
        records: List[Dict] = []
        for m in self._FUNC_RE.finditer(content):
            name = m.group(1)
            # Skip common false positives
            if name in {"if", "for", "while", "switch", "catch"}:
                continue
            start_line = content[: m.start()].count("\n")
            end_line = len(lines) - 1   # default: rest of file
            # scan forward tracking brace depth to find the real closing brace
            depth = 0
            for i, line in enumerate(lines[start_line:], start=start_line):
                depth += line.count("{") - line.count("}")
                if depth == 0 and i > start_line:
                    end_line = i
                    break
            code = "\n".join(lines[start_line : end_line + 1])
            records.append(
                self._make_record(
                    code=code,
                    name=name,
                    lang="cpp",
                    fpath=fpath,
                    repo_name=repo_name,
                    repo_root=repo_root,
                    line_start=start_line + 1,
                    line_end=end_line + 1,
                    dfg=[],
                )
            )
        return records

    # ── shared record builder ─────────────────────────────────────────
    @staticmethod
    def _make_record(
        code: str,
        name: str,
        lang: str,
        fpath: Path,
        repo_name: str,
        repo_root: Path,
        line_start: int,
        line_end: int,
        dfg: List,
    ) -> Dict:
        tokens = _simple_tokenize(code)
        return {
            "idx": str(uuid.uuid4()),
            "repo": repo_name,
            "path": str(fpath.relative_to(repo_root)),
            "func_name": name,
            "original_string": code,
            "language": lang,
            "code": code,
            "code_tokens": tokens,
            "docstring": "",
            "docstring_tokens": [],
            "sha": "unknown",
            "url": "",
            "partition": "train",
            "variable_positions": [],
            "dataflow_graph": dfg,
            "line_start": line_start,
            "line_end": line_end,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 – Erlang extractor (wraps your existing parser when available)
# ═══════════════════════════════════════════════════════════════════════════

ERL_EXTENSIONS = {".erl", ".hrl"}


def _extract_erl_docstring(content: str, line_start: int) -> str:
    """Extract %% comment block immediately above line_start (1-indexed)."""
    lines = content.splitlines()
    doc_lines = []
    # Walk backwards from the line just before the function
    for ln in reversed(lines[: line_start - 1]):
        stripped = ln.strip()
        if stripped.startswith("%%"):
            doc_lines.insert(0, stripped.lstrip("% ").strip())
        elif stripped == "" and not doc_lines:
            # allow one blank line between comment and function
            continue
        else:
            break
    return " ".join(doc_lines)

# Regex for the fallback path (no tree-sitter-erlang / ErlangParser available)
_ERL_FUNC_RE = re.compile(
    r"""
    (?:^%-+.*\n)*           # optional comment block before the function
    ^([a-z_]\w*)            # function name  (group 1)
    \s*\(                   # opening paren
    """,
    re.VERBOSE | re.MULTILINE,
)


class ErlangExtractor:
    """
    Wraps your existing ErlangParser when it is importable; otherwise uses a
    lightweight regex fallback so the script works in any environment.
    """

    def __init__(self, ts_parser=None, ts_langs: dict = None):
        self._ParserClass = ts_parser  # Parser class, passed through to _build_dfg
        self._ts_parser = ts_parser
        self._ts_langs = ts_langs or {}
        self._erl_parser = None

        # Make sure the directory that *contains* parsers/ is on sys.path.
        # We add both the script's own directory and the current working directory
        # so it works whether you cd into Evaluation/ or call the script from outside.
        for _candidate in (Path(__file__).resolve().parent, Path.cwd()):
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))

        # Try to import the project-local ErlangParser
        try:
            from parsers.erlang_parser import ErlangParser  # type: ignore
            self._erl_parser = ErlangParser()
            log.info("ErlangParser loaded from parsers.erlang_parser")
        except Exception as _erl_exc:
            import traceback as _tb
            log.info(
                "parsers.erlang_parser not available – using regex fallback for Erlang.\n"
                "  Full traceback:\n%s",
                "".join(_tb.format_exception(_erl_exc)).rstrip(),
            )

    # ------------------------------------------------------------------
    def extract_from_repo(self, repo_path: str, repo_name: str) -> List[Dict]:
        repo_path = Path(repo_path)
        records: List[Dict] = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in ERL_EXTENSIONS:
                    continue
                try:
                    records.extend(
                        self._extract_from_file(fpath, repo_name, repo_path)
                    )
                except Exception as exc:
                    log.warning("Skipping %s: %s", fpath, exc)
        log.info("Erlang extractor: %d functions from %s", len(records), repo_name)
        return records

    # ------------------------------------------------------------------
    def _extract_from_file(
        self, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if self._erl_parser is not None:
            return self._extract_native(content, fpath, repo_name, repo_root)
        result = self._extract_regex(content, fpath, repo_name, repo_root)
        log.info("  %s → %d functions (regex)", fpath.name, len(result))
        return result

    # ── native ErlangParser path ──────────────────────────────────────
    def _extract_native(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        """Delegate to the existing ErlangParser and convert to unified format."""
        p = self._erl_parser
        root_node = p.parse_string(content)
        if root_node is None:
            log.debug("parse_string returned None for %s", fpath)
            return []

        func_nodes = p.extract_functions(root_node)
        log.info("  %s → %d functions", fpath.name, len(func_nodes))

        records: List[Dict] = []
        for func_node in func_nodes:
            try:
                name = p.get_function_name(func_node) or "<unknown>"
                code = p.node_text(func_node)
                if not code.strip():
                    continue

                # Tokens + variable info via the real GraphCodeBERT extraction API
                tokens, var_indices, var_names, is_clause_params = p.extract_graphcodebert_data(func_node, [])
                dfg = p.create_dataflow_graph(var_indices, var_names, is_clause_params)

                # variable_positions in the (token_index, var_name, is_clause_param) format
                var_positions = list(zip(var_indices, var_names, is_clause_params))

                start = func_node.start_point[0] + 1
                end = func_node.end_point[0] + 1

                # Extract %% comment block sitting directly above the function
                docstring = _extract_erl_docstring(content, start)
            except Exception as exc:
                log.debug("Function extraction error in %s: %s", fpath, exc)
                continue

            records.append({
                "idx": str(uuid.uuid4()),
                "repo": repo_name,
                "path": str(fpath.relative_to(repo_root)),
                "func_name": name,
                "original_string": code,
                "language": "erlang",
                "code": code,
                "code_tokens": tokens,
                "docstring": docstring or "",
                "docstring_tokens": docstring.split() if docstring else [],
                "sha": "unknown",
                "url": "",
                "partition": "train",
                "variable_positions": var_positions,
                "dataflow_graph": dfg,
                "line_start": start,
                "line_end": end,
            })
        return records

    # ── regex fallback ────────────────────────────────────────────────
    def _extract_regex(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        """
        Naive but robust: split file at function boundaries using the
        'name(args) ->' pattern and capture until the next top-level function.
        """
        # Split on top-level function clause boundaries
        boundaries = [m.start() for m in _ERL_FUNC_RE.finditer(content)]
        boundaries.append(len(content))
        records: List[Dict] = []
        lines = content.splitlines()

        for i in range(len(boundaries) - 1):
            snippet = content[boundaries[i] : boundaries[i + 1]].strip()
            if not snippet or len(snippet) < 20:
                continue
            m = _ERL_FUNC_RE.match(snippet)
            name = m.group(1) if m else "<unknown>"
            # Skip Erlang attribute lines (-module, -export, …)
            if snippet.startswith("-"):
                continue
            start_line = content[: boundaries[i]].count("\n") + 1
            end_line = start_line + snippet.count("\n")

            # Grab preceding comment as docstring
            docstring = ""
            doc_lines = []
            for ln in reversed(lines[: start_line - 1]):
                stripped = ln.strip()
                if stripped.startswith("%"):
                    doc_lines.insert(0, stripped.lstrip("% "))
                else:
                    break
            docstring = " ".join(doc_lines)

            dfg = _build_dfg(snippet, "erlang", self._ts_parser, self._ts_langs)
            records.append({
                "idx": str(uuid.uuid4()),
                "repo": fpath.stem,
                "path": str(fpath.relative_to(fpath.parent.parent) if fpath.parent != fpath.parent.parent else fpath.name),
                "func_name": name,
                "original_string": snippet,
                "language": "erlang",
                "code": snippet,
                "code_tokens": _simple_tokenize(snippet),
                "docstring": docstring,
                "docstring_tokens": docstring.split(),
                "sha": "unknown",
                "url": "",
                "partition": "train",
                "variable_positions": [],
                "dataflow_graph": dfg,
                "line_start": start_line,
                "line_end": end_line,
            })
        return records


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4b – Python extractor
# ═══════════════════════════════════════════════════════════════════════════

PY_EXTENSIONS = {".py"}

_PY_FUNC_RE = re.compile(
    r"""
    ^(?P<indent>[ \t]*)          # leading indent (captures method indentation)
    (?:async\s+)?def\s+          # optional async, then def
    (?P<name>[A-Za-z_]\w*)       # function / method name
    \s*\(                        # opening paren
    """,
    re.VERBOSE | re.MULTILINE,
)


def _extract_py_docstring(lines: List[str], body_start: int) -> str:
    """Return the first docstring found at the top of a function body (if any)."""
    # body_start is the 0-indexed line of the 'def' statement; body is +1 onward
    i = body_start + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        return ""
    stripped = lines[i].strip()
    # Single-line docstring
    for q in ('"""', "'''", '"', "'"):
        if stripped.startswith(q) and stripped.endswith(q) and len(stripped) > 2 * len(q):
            return stripped[len(q):-len(q)]
    # Multi-line docstring
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            doc_lines = [stripped[len(q):]]
            i += 1
            while i < len(lines):
                l = lines[i]
                if q in l:
                    doc_lines.append(l[:l.index(q)])
                    break
                doc_lines.append(l)
                i += 1
            return " ".join(ln.strip() for ln in doc_lines if ln.strip())
    return ""


class PythonExtractor:
    """
    Extracts top-level functions and class methods from Python source files.

    Uses tree-sitter (tree_sitter_python) when available; otherwise falls back
    to an indent-aware regex heuristic.
    """

    def __init__(self, ts_parser=None, ts_langs: dict = None):
        self._ParserClass = ts_parser
        self._ts_langs = ts_langs or {}
        self._use_ts = ts_parser is not None and "python" in self._ts_langs

    # ------------------------------------------------------------------
    def extract_from_repo(self, repo_path: str, repo_name: str) -> List[Dict]:
        repo_path = Path(repo_path)
        records: List[Dict] = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in PY_EXTENSIONS:
                    continue
                try:
                    records.extend(
                        self._extract_from_file(fpath, repo_name, repo_path)
                    )
                except Exception as exc:
                    log.warning("Skipping %s: %s", fpath, exc)
        log.info("Python extractor: %d functions from %s", len(records), repo_name)
        return records

    # ------------------------------------------------------------------
    def _extract_from_file(
        self, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if self._use_ts:
            return self._extract_ts(content, fpath, repo_name, repo_root)
        result = self._extract_regex(content, fpath, repo_name, repo_root)
        log.info("  %s → %d functions (regex)", fpath.name, len(result))
        return result

    # ── tree-sitter path ──────────────────────────────────────────────
    def _extract_ts(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        parser = self._ParserClass(self._ts_langs["python"])
        tree = parser.parse(content.encode("utf-8"))
        lines = content.splitlines()
        records: List[Dict] = []
        code_bytes = content.encode("utf-8")

        def visit(node):
            if node.type in ("function_definition", "decorated_definition"):
                # For decorated_definition, dig into the actual function_definition
                target = node
                if node.type == "decorated_definition":
                    for child in node.children:
                        if child.type == "function_definition":
                            target = child
                            break
                    else:
                        for child in node.children:
                            visit(child)
                        return

                try:
                    start = node.start_point[0]   # include decorators
                    end = target.end_point[0]
                    code = "\n".join(lines[start: end + 1])

                    # Name: first identifier child of the function_definition
                    name = "<unknown>"
                    for child in target.children:
                        if child.type == "identifier":
                            name = code_bytes[child.start_byte: child.end_byte].decode("utf-8", "ignore")
                            break

                    docstring = _extract_py_docstring(lines, target.start_point[0])
                    dfg = _build_dfg(code, "python", self._ParserClass, self._ts_langs)

                    records.append(self._make_record(
                        code=code,
                        name=name,
                        fpath=fpath,
                        repo_name=repo_name,
                        repo_root=repo_root,
                        line_start=start + 1,
                        line_end=end + 1,
                        docstring=docstring,
                        dfg=dfg,
                    ))
                except Exception as exc:
                    log.debug("Python ts extraction error in %s: %s", fpath, exc)

            # Always recurse so we pick up methods inside classes
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        log.info("  %s → %d functions (tree-sitter)", fpath.name, len(records))
        return records

    # ── regex fallback path ───────────────────────────────────────────
    def _extract_regex(
        self, content: str, fpath: Path, repo_name: str, repo_root: Path
    ) -> List[Dict]:
        lines = content.splitlines()
        records: List[Dict] = []

        for m in _PY_FUNC_RE.finditer(content):
            name = m.group("name")
            indent = m.group("indent")
            def_line = content[: m.start()].count("\n")  # 0-indexed

            # Collect the body: lines that are either more indented than def_line
            # or blank, until we hit a line at the same/lesser indentation.
            end_line = def_line
            for i, line in enumerate(lines[def_line + 1:], start=def_line + 1):
                if line.strip() == "":
                    end_line = i
                    continue
                if len(line) - len(line.lstrip()) > len(indent):
                    end_line = i
                else:
                    break

            code = "\n".join(lines[def_line: end_line + 1])
            docstring = _extract_py_docstring(lines, def_line)
            dfg = _build_dfg(code, "python", self._ParserClass, self._ts_langs)

            records.append(self._make_record(
                code=code,
                name=name,
                fpath=fpath,
                repo_name=repo_name,
                repo_root=repo_root,
                line_start=def_line + 1,
                line_end=end_line + 1,
                docstring=docstring,
                dfg=dfg,
            ))
        return records

    # ── shared record builder ─────────────────────────────────────────
    @staticmethod
    def _make_record(
        code: str,
        name: str,
        fpath: Path,
        repo_name: str,
        repo_root: Path,
        line_start: int,
        line_end: int,
        docstring: str,
        dfg: List,
    ) -> Dict:
        tokens = _simple_tokenize(code)
        return {
            "idx": str(uuid.uuid4()),
            "repo": repo_name,
            "path": str(fpath.relative_to(repo_root)),
            "func_name": name,
            "original_string": code,
            "language": "python",
            "code": code,
            "code_tokens": tokens,
            "docstring": docstring,
            "docstring_tokens": docstring.split() if docstring else [],
            "sha": "unknown",
            "url": "",
            "partition": "train",
            "variable_positions": [],
            "dataflow_graph": dfg,
            "line_start": line_start,
            "line_end": line_end,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 – INDEX mode
# ═══════════════════════════════════════════════════════════════════════════

def cmd_index(args: argparse.Namespace) -> None:
    """
    Parse repos → JSONL corpus → (optionally) compute + save embeddings.

    The embedding index is only built when --model is also supplied.
    Without --model the script still produces a valid JSONL that can be
    indexed later with `search` mode.
    """
    has_specific = args.erlang or args.cpp or args.python
    has_repo = bool(args.repo)

    if not has_specific and not has_repo:
        log.error(
            "Provide at least one source: --repo, --erlang, --cpp, or --python."
        )
        sys.exit(1)

    # ── optional tree-sitter setup ────────────────────────────────────
    ts_parser, ts_langs = _try_load_tree_sitter()

    # ── build language→path lists ─────────────────────────────────────
    # --repo auto-discovers all three languages under a single root;
    # language-specific flags are additive on top of that.
    erlang_paths: List[str] = list(args.erlang or [])
    cpp_paths: List[str]    = list(args.cpp or [])
    python_paths: List[str] = list(args.python or [])

    for repo_root in args.repo or []:
        erlang_paths.append(repo_root)
        cpp_paths.append(repo_root)
        python_paths.append(repo_root)

    # ── extract functions ─────────────────────────────────────────────
    all_records: List[Dict] = []

    for erl_path in erlang_paths:
        ext = ErlangExtractor(ts_parser=ts_parser, ts_langs=ts_langs)
        repo_name = Path(erl_path).name
        all_records.extend(ext.extract_from_repo(erl_path, repo_name))

    for cpp_path in cpp_paths:
        ext = CppExtractor(ts_parser=ts_parser, ts_langs=ts_langs)
        repo_name = Path(cpp_path).name
        all_records.extend(ext.extract_from_repo(cpp_path, repo_name))

    for py_path in python_paths:
        ext = PythonExtractor(ts_parser=ts_parser, ts_langs=ts_langs)
        repo_name = Path(py_path).name
        all_records.extend(ext.extract_from_repo(py_path, repo_name))

    if not all_records:
        log.error("No functions extracted. Check repo paths and file extensions.")
        sys.exit(1)

    if args.limit and len(all_records) > args.limit:
        log.info("Limiting to %d functions (--limit %d, total extracted: %d)",
                 args.limit, args.limit, len(all_records))
        all_records = all_records[: args.limit]

    log.info("Total functions to index: %d", len(all_records))

    # ── write JSONL ───────────────────────────────────────────────────
    out_jsonl = Path(args.output)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Corpus written to %s  (%d entries)", out_jsonl, len(all_records))

    # ── optionally build embedding index ──────────────────────────────
    if args.model:
        _build_index(all_records, args.model, Path(args.index), args.batch_size)
    else:
        log.info(
            "No --model supplied; skipping embedding index. "
            "Run `search` later with --model and --jsonl to build it on demand."
        )


def _try_load_tree_sitter():
    """Return (parser_instance, lang_dict) or (None, {}) if tree-sitter is not installed.

    Supports both old tree-sitter API (set_language) and new API (Parser(language)).
    """
    try:
        from tree_sitter import Parser, Language
        langs: Dict[str, Any] = {}
        try:
            import tree_sitter_cpp as tscpp
            langs["cpp"] = Language(tscpp.language())
        except ImportError:
            pass
        try:
            import tree_sitter_erlang as tserl
            langs["erlang"] = Language(tserl.language())
        except ImportError:
            pass
        try:
            import tree_sitter_python as tspy
            langs["python"] = Language(tspy.language())
        except ImportError:
            pass
        if langs:
            log.info("tree-sitter available for: %s", list(langs))
        else:
            log.info("tree-sitter installed but no language bindings found; using regex fallback")
        # Return the Parser CLASS — callers do Parser(language) per file (>=0.22 API)
        return Parser, langs
    except ImportError:
        log.info("tree-sitter not available; using regex fallback")
        return None, {}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 – Embedding helpers (shared by index + search)
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_and_tokenizer(model_path: str):
    """Load the encoder saved by CodeSearchModel.save_pretrained().

    The training script saves only model.encoder (an AutoModel) via
    save_pretrained(), so best_model/ is a plain transformer encoder —
    no LM head, no wrapper. We load it with AutoModel and use the [CLS]
    token as the embedding vector, exactly as the CodeSearchModel does
    at inference time.

    Tokenizer resolution:
      1. Local model directory (offline, no network call).
      2. HF Hub as fallback (requires internet).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Tokenizer — try local first
    if Path(model_path).is_dir() and Path(model_path).joinpath("vocab.json").exists():
        log.info("Loading tokenizer from local model directory: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        _hub_id = "microsoft/graphcodebert-base"
        log.info("vocab.json not found locally – loading tokenizer from HF Hub: %s", _hub_id)
        tokenizer = AutoTokenizer.from_pretrained(_hub_id)

    # Encoder — the checkpoint IS the encoder (model.encoder.save_pretrained)
    log.info("Loading encoder from %s …", model_path)
    model = AutoModel.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    log.info("Model on %s", device)
    return tokenizer, model, device


def _mean_pool(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings, masking out padding.

    Must match model.py's mean_pooling() exactly — the model was trained
    with mean pooling (+ L2 normalization), not CLS-token pooling.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _embed_records(
    records: List[Dict],
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 256,
) -> "torch.Tensor":
    """
    Encode *code only* for every record and return a float32 tensor of shape
    (N, hidden_size), mean-pooled and L2-normalized.

    Training (train.py) never concatenated docstring + code into one
    sequence — it ran the shared encoder separately on `code` (truncated to
    code_len=256) to get the code-branch vector, and separately on a
    docstring (truncated to nl_len=128) to get the NL-branch vector, then
    compared the two via dot product. To match that, indexing here embeds
    `code` alone, at the same 256-token length used in training.
    """
    import torch
    import torch.nn.functional as F

    all_embeddings: List["torch.Tensor"] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [r.get("code") or "" for r in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())
        log.info(
            "Embedded %d / %d records",
            min(start + batch_size, len(records)),
            len(records),
        )
    return torch.cat(all_embeddings, dim=0)


def _build_index(
    records: List[Dict], model_path: str, index_path: Path, batch_size: int
) -> None:
    import torch

    tokenizer, model, device = _load_model_and_tokenizer(model_path)
    embeddings = _embed_records(records, tokenizer, model, device, batch_size)
    # Already L2-normalized inside _embed_records — no extra normalization needed.

    index_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings}, str(index_path))
    log.info("Embedding index saved to %s  shape=%s", index_path, list(embeddings.shape))


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 – SEARCH mode
# ═══════════════════════════════════════════════════════════════════════════

def _load_corpus_and_index(args: argparse.Namespace):
    """
    Shared loader used by both `search` and `eval`.

    Loads the JSONL corpus, the fine-tuned model/tokenizer, and the embedding
    index (building + caching it on demand if it does not exist yet).

    Returns (records, embeddings, tokenizer, model, device).
    """
    import torch

    jsonl_path = Path(args.jsonl)
    index_path = Path(args.index)

    if not jsonl_path.exists():
        log.error("JSONL file not found: %s", jsonl_path)
        sys.exit(1)

    # ── load corpus ───────────────────────────────────────────────────
    log.info("Loading corpus from %s …", jsonl_path)
    records: List[Dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("%d records loaded", len(records))

    # ── load or build embedding index ─────────────────────────────────
    tokenizer, model, device = _load_model_and_tokenizer(args.model)

    if index_path.exists():
        log.info("Loading embedding index from %s …", index_path)
        data = torch.load(str(index_path), map_location="cpu")
        embeddings = data["embeddings"]
    else:
        log.info("Index not found – building now (this may take a while) …")
        embeddings = _embed_records(records, tokenizer, model, device, args.batch_size)
        # Already L2-normalized inside _embed_records — no extra normalization needed.
        torch.save({"embeddings": embeddings}, str(index_path))
        log.info("Index saved to %s", index_path)

    embeddings = embeddings.to(device)
    return records, embeddings, tokenizer, model, device


def cmd_search(args: argparse.Namespace) -> None:
    """
    Interactive semantic code search.

    On first run (or when --index does not exist) the embedding index is built
    from the JSONL and saved for future use.
    """
    records, embeddings, tokenizer, model, device = _load_corpus_and_index(args)

    # ── interactive search loop ───────────────────────────────────────
    top_k = args.top
    print("\n" + "═" * 60)
    print("  Code Search  —  GraphCodeBERT semantic search")
    print(f"  Corpus: {len(records)} functions  |  top-{top_k} results")
    print("  Type a query and press Enter.  Ctrl-C or 'quit' to exit.")
    print("═" * 60 + "\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        hits = _search(query, embeddings, records, tokenizer, model, device, top_k)
        _print_hits(hits)


def _search(
    query: str,
    embeddings: "torch.Tensor",
    records: List[Dict],
    tokenizer,
    model,
    device,
    top_k: int,
) -> List[Tuple[float, Dict]]:
    """
    Encode the query and return the top-k (score, record) pairs.

    The query is treated as the NL branch (same as docstrings at training
    time): mean-pooled, L2-normalized, truncated to nl_len=128.
    """
    import torch
    import torch.nn.functional as F

    enc = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    q_vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    q_vec = F.normalize(q_vec, p=2, dim=1)

    scores = (embeddings @ q_vec.T).squeeze(1)   # (N,)
    top_indices = scores.topk(min(top_k, len(records))).indices.tolist()

    return [(float(scores[i]), records[i]) for i in top_indices]


def _print_hits(hits: List[Tuple[float, Dict]]) -> None:
    print()
    for rank, (score, rec) in enumerate(hits, start=1):
        lang = rec.get("language", "?")
        name = rec.get("func_name", "<unknown>")
        repo = rec.get("repo", "")
        path = rec.get("path", "")
        line_start = rec.get("line_start", "?")
        line_end = rec.get("line_end", "?")
        doc = (rec.get("docstring") or "").strip().replace("\n", " ")
        code_preview = rec.get("code", "").strip()

        # Trim code preview to first 8 lines
        preview_lines = code_preview.splitlines()[:8]
        preview = "\n    ".join(preview_lines)
        if len(code_preview.splitlines()) > 8:
            preview += "\n    …"

        print(f"  ┌─ #{rank}  score={score:.4f}  [{lang}]  {repo}/{path}:{line_start}-{line_end}")
        print(f"  │  func: {name}")
        if doc:
            print(f"  │  doc:  {doc[:120]}")
        print(f"  │")
        print(f"  │  {preview}")
        print(f"  └{'─'*60}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7b – EVAL mode (batch queries against a query/good-answer dataset)
# ═══════════════════════════════════════════════════════════════════════════
#
# Eval dataset format (one JSON object per line, .jsonl):
#   {
#     "question": "...",
#     "expectedTypes": {...},                 # unused for now
#     "results": {
#         "record:<path>:<Struct>.<field>": <relevance 1-5>,
#         "function:<path>:<func_name>/<arity>": <relevance 1-5>,
#         "tr:<TICKET-ID>": <relevance 1-5>,
#         "mom:<Something>::<field>": <relevance 1-5>,
#         ...
#     }
#   }
#
# NOTE: for now we only score against the "function:" entries in `results`
# (per current scope) — "record:", "tr:", and "mom:" entries are parsed but
# ignored. This can be extended later once those doc types have a
# corresponding representation in the indexed corpus.
#
# Matching is done on (path, func_name); the /<arity> suffix is dropped since
# the corpus schema built by `index` mode doesn't track arity, so a function
# with multiple arities in the same file is treated as a single doc. This is
# a pragmatic simplification — flag if you need arity-aware matching later.

def _rank_and_score_all(
    query: str,
    embeddings: "torch.Tensor",
    tokenizer,
    model,
    device,
):
    """Encode *query* and return a 1-D tensor of similarity scores, one per
    corpus record, in the same order as `embeddings`."""
    import torch
    import torch.nn.functional as F

    enc = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    q_vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    q_vec = F.normalize(q_vec, p=2, dim=1)
    return (embeddings @ q_vec.T).squeeze(1).cpu()  # (N,)


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").strip().lstrip("./")


def _parse_function_gold_key(raw_key: str):
    """
    Parse a "function:<path>:<func_name>/<arity>" results key.
    Returns (path, func_name) or None if it doesn't match the expected shape.
    """
    if not raw_key.startswith("function:"):
        return None
    rest = raw_key[len("function:"):]
    if ":" not in rest:
        return None
    path, func_spec = rest.rsplit(":", 1)
    func_name = func_spec.rsplit("/", 1)[0] if "/" in func_spec else func_spec
    return _norm_path(path), func_name


def _load_function_eval_dataset(path: Path, min_relevance: int = 1) -> List[Dict]:
    """
    Load the {"question", "expectedTypes", "results"} JSONL eval format and
    reduce each row's `results` to only its "function:" entries, keyed by
    normalized (path, func_name).

    min_relevance: drop function-type gold entries with relevance below this
    (e.g. min_relevance=5 keeps only entries explicitly graded as the single
    correct answer; min_relevance=1 keeps every graded function entry).

    Returns a list of {"query_id", "question", "gold": {(path, func_name): relevance}}.
    """
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataset = []
    total_gold_entries = 0
    dropped_below_threshold = 0
    for i, row in enumerate(rows):
        question = row.get("question")
        results = row.get("results", {})
        if not question or not isinstance(results, dict):
            log.warning("Skipping row %d — missing 'question' or 'results'.", i)
            continue

        gold: Dict[Tuple[str, str], int] = {}
        for raw_key, relevance in results.items():
            parsed = _parse_function_gold_key(raw_key)
            if parsed is None:
                continue  # not a "function:" entry — ignored for now
            total_gold_entries += 1
            if relevance < min_relevance:
                dropped_below_threshold += 1
                continue
            path_key, func_name = parsed
            # Keep the max relevance if the same (path, func_name) appears twice.
            gold[(path_key, func_name)] = max(relevance, gold.get((path_key, func_name), 0))

        dataset.append({"query_id": f"q{i}", "question": question, "gold": gold})

    n_with_gold = sum(1 for d in dataset if d["gold"])
    kept = total_gold_entries - dropped_below_threshold
    log.info(
        "Loaded %d queries (%d have at least one 'function:' gold entry at or above "
        "min_relevance=%d; %d/%d total function-type gold entries kept, %d dropped "
        "for being below the threshold).",
        len(dataset), n_with_gold, min_relevance, kept, total_gold_entries, dropped_below_threshold,
    )
    return dataset


def _build_function_doc_index(records: List[Dict]) -> Dict[Tuple[str, str], List[int]]:
    """Map normalized (path, func_name) -> list of record indices sharing it."""
    idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = (_norm_path(rec.get("path", "")), rec.get("func_name", ""))
        idx[key].append(i)
    return idx


def _resolve_gold_doc_ids(
    gold: Dict[Tuple[str, str], int],
    doc_index: Dict[Tuple[str, str], List[int]],
) -> Tuple[Dict[str, int], int]:
    """
    Resolve (path, func_name) gold keys to doc ids that exist in the corpus.

    Tries an exact (path, func_name) match first; falls back to a path
    suffix match (handles cases where the dataset's path root differs
    slightly from the corpus's) if the exact match fails.

    Returns (resolved {doc_id: relevance}, count_unresolved).
    """
    # Build a func_name -> [(corpus_path, doc_id)] lookup lazily for fallback.
    resolved: Dict[str, int] = {}
    unresolved = 0

    # Fallback index: func_name -> list of (corpus_path, key)
    by_func_name: Dict[str, List[Tuple[str, Tuple[str, str]]]] = defaultdict(list)
    for key in doc_index:
        by_func_name[key[1]].append((key[0], key))

    for (gold_path, func_name), relevance in gold.items():
        key = (gold_path, func_name)
        match_key = None
        if key in doc_index:
            match_key = key
        else:
            # Fallback: same func_name, path is a suffix/prefix match.
            for corpus_path, candidate_key in by_func_name.get(func_name, []):
                if corpus_path.endswith(gold_path) or gold_path.endswith(corpus_path):
                    match_key = candidate_key
                    break

        if match_key is None:
            unresolved += 1
            continue

        doc_id = f"{match_key[0]}::{match_key[1]}"
        resolved[doc_id] = max(relevance, resolved.get(doc_id, 0))

    return resolved, unresolved


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Batch-evaluate the model against a query/good-answer dataset, scoring
    against the "function:" entries only, using ranx for IR metrics.
    """
    try:
        from ranx import Qrels, Run, evaluate as ranx_evaluate
    except ImportError:
        log.error(
            "The 'ranx' package is required for eval mode. Install it with:\n"
            "    pip install ranx --break-system-packages"
        )
        sys.exit(1)

    records, embeddings, tokenizer, model, device = _load_corpus_and_index(args)
    doc_index = _build_function_doc_index(records)

    # doc_id -> list of record indices (for scoring: take max score if a
    # (path, func_name) maps to more than one record, e.g. distinct arities).
    doc_id_to_indices: Dict[str, List[int]] = {
        f"{k[0]}::{k[1]}": idxs for k, idxs in doc_index.items()
    }

    queries_path = Path(args.queries)
    if not queries_path.exists():
        log.error("Eval dataset not found: %s", queries_path)
        sys.exit(1)
    dataset = _load_function_eval_dataset(queries_path, min_relevance=args.min_relevance)
    if args.limit:
        dataset = dataset[: args.limit]

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    qrels_dict: Dict[str, Dict[str, int]] = {}
    run_dict: Dict[str, Dict[str, float]] = {}
    per_query_rows = []
    total_unresolved = 0
    skipped_no_gold = 0

    for row in dataset:
        resolved_gold, unresolved = _resolve_gold_doc_ids(row["gold"], doc_index)
        total_unresolved += unresolved

        if not resolved_gold:
            skipped_no_gold += 1
            per_query_rows.append({"query_id": row["query_id"], "question": row["question"], "note": "no resolvable function gold"})
            continue

        scores = _rank_and_score_all(row["question"], embeddings, tokenizer, model, device)
        run_scores: Dict[str, float] = {}
        for doc_id, idxs in doc_id_to_indices.items():
            run_scores[doc_id] = max(float(scores[i]) for i in idxs)

        qrels_dict[row["query_id"]] = resolved_gold
        run_dict[row["query_id"]] = run_scores
        per_query_rows.append({"query_id": row["query_id"], "question": row["question"], "gold": resolved_gold})

    evaluated = len(qrels_dict)
    print("\n" + "═" * 60)
    print("  Code Search — Evaluation Results (function-type gold only)")
    print(f"  Corpus: {len(records)} functions  |  min_relevance={args.min_relevance}")
    print(f"  Queries: {len(dataset)} total, {evaluated} evaluated, "
          f"{skipped_no_gold} skipped (no 'function:' gold resolvable in corpus)")
    if total_unresolved:
        print(f"  Note: {total_unresolved} individual 'function:' gold references across "
              f"all queries could not be matched to a corpus record (path/name mismatch?).")
    print("═" * 60)

    if evaluated:
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        results = ranx_evaluate(qrels, run, metrics=metrics)
        for m in metrics:
            print(f"  {m:<12s}: {results[m]:.4f}")
    else:
        print("  No queries could be evaluated — check path/func_name matching above.")
    print("═" * 60 + "\n")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in per_query_rows:
                fh.write(json.dumps(r) + "\n")
        log.info("Per-query results written to %s", out_path)


#  SECTION 8 – CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── index ──────────────────────────────────────────────────────────
    idx = sub.add_parser(
        "index",
        help="Parse repos and write a JSONL corpus (+ optional embedding index).",
    )
    idx.add_argument(
        "--repo",
        metavar="REPO_PATH",
        nargs="+",
        help="Path(s) to a multilingual repository. Erlang, C++, and Python "
             "files are all discovered automatically.",
    )
    idx.add_argument(
        "--erlang",
        metavar="REPO_PATH",
        nargs="+",
        help="Path(s) to locally cloned Erlang repositories.",
    )
    idx.add_argument(
        "--cpp",
        metavar="REPO_PATH",
        nargs="+",
        help="Path(s) to locally cloned C++ repositories.",
    )
    idx.add_argument(
        "--python",
        metavar="REPO_PATH",
        nargs="+",
        help="Path(s) to locally cloned Python repositories.",
    )
    idx.add_argument(
        "--output",
        metavar="FILE",
        default="corpus.jsonl",
        help="Output JSONL file path (default: corpus.jsonl).",
    )
    idx.add_argument(
        "--index",
        metavar="FILE",
        default="index.pt",
        help="Output embedding index path (default: index.pt).",
    )
    idx.add_argument(
        "--model",
        metavar="DIR",
        default=None,
        help="Fine-tuned model path. When given, embeddings are computed now.",
    )
    idx.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32).",
    )
    idx.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Cap total extracted functions (useful for quick tests).",
    )

    # ── search ─────────────────────────────────────────────────────────
    srch = sub.add_parser(
        "search",
        help="Interactive semantic code search against a pre-built corpus.",
    )
    srch.add_argument(
        "--model",
        metavar="DIR",
        required=True,
        help="Fine-tuned GraphCodeBERT model directory.",
    )
    srch.add_argument(
        "--jsonl",
        metavar="FILE",
        default="corpus.jsonl",
        help="Corpus JSONL file (default: corpus.jsonl).",
    )
    srch.add_argument(
        "--index",
        metavar="FILE",
        default="index.pt",
        help="Embedding index .pt file (default: index.pt). Built on demand.",
    )
    srch.add_argument(
        "--top",
        metavar="N",
        type=int,
        default=5,
        help="Number of results to return per query (default: 5).",
    )
    srch.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=32,
        help="Batch size when building index on demand (default: 32).",
    )

    # ── eval ───────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "eval",
        help="Batch-evaluate the model against a query/good-answer dataset "
             "(reports Recall@k and MRR; no interactive prompt).",
    )
    ev.add_argument(
        "--model",
        metavar="DIR",
        required=True,
        help="Fine-tuned GraphCodeBERT model directory.",
    )
    ev.add_argument(
        "--jsonl",
        metavar="FILE",
        default="corpus.jsonl",
        help="Corpus JSONL file (default: corpus.jsonl).",
    )
    ev.add_argument(
        "--index",
        metavar="FILE",
        default="index.pt",
        help="Embedding index .pt file (default: index.pt). Built on demand.",
    )
    ev.add_argument(
        "--queries",
        metavar="FILE",
        required=True,
        help="Query/gold dataset (.jsonl), one JSON object per line: "
             '{"question": ..., "results": {"function:<path>:<name>/<arity>": <relevance>, ...}}. '
             "Only 'function:' entries in results are scored currently.",
    )
    ev.add_argument(
        "--metrics",
        metavar="M1,M2,...",
        default="mrr,recall@5,recall@10,ndcg@5,ndcg@10,precision@5",
        help="Comma-separated ranx metric names (default: "
             "mrr,recall@5,recall@10,ndcg@5,ndcg@10,precision@5).",
    )
    ev.add_argument(
        "--min-relevance",
        metavar="N",
        type=int,
        default=1,
        help="Drop function-type gold entries graded below this (1-5) before "
             "scoring. Default 1 keeps every graded function entry. Use 5 for "
             "a strict 'only the single correct answer counts' mode — note "
             "this may skip many queries whose graded-5 answer is a "
             "record/tr/mom entry rather than a function.",
    )
    ev.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Only evaluate the first N queries (useful for a quick smoke test).",
    )
    ev.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Optional path to write per-query rank results as JSONL.",
    )
    ev.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=32,
        help="Batch size when building index on demand (default: 32).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()