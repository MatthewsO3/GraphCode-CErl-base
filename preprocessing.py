import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from datasets import load_dataset
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
from tqdm import tqdm


LANGUAGE_CONFIG = {
    "cpp": {
        "lib": "tree_sitter_cpp",
        "def_nodes": ["declaration", "init_declarator", "parameter_declaration"],
        "assignment_nodes": ["assignment_expression"],
        "identifier_nodes": ["identifier", "field_identifier"],
        "keywords": ["void ", "int ", "class ", "std::"],
        "dataset_name": "C++-all",
    },
    "python": {
        "lib": "tree_sitter_python",
        "def_nodes": [
            "assignment",
            "for_statement",
            "function_definition",
            "parameters",
        ],
        "assignment_nodes": ["augmented_assignment"],
        "identifier_nodes": ["identifier"],
        "keywords": ["def ", "import ", "class ", "self."],
        "dataset_name": "Python-all",
    },
    "java": {
        "lib": "tree_sitter_java",
        "def_nodes": [
            "local_variable_declaration",
            "formal_parameter",
            "method_declaration",
        ],
        "assignment_nodes": ["assignment_expression"],
        "identifier_nodes": ["identifier"],
        "keywords": ["public ", "private ", "class ", "String "],
        "dataset_name": "Java-all",
    },
    "javascript": {
        "lib": "tree_sitter_javascript",
        "def_nodes": [
            "variable_declarator",
            "formal_parameters",
            "function_declaration",
        ],
        "assignment_nodes": ["assignment_expression"],
        "identifier_nodes": ["identifier", "variable"],
        "keywords": ["function ", "const ", "let ", "var ", "=>"],
        "dataset_name": "JavaScript-all",
    },
}


class UniversalPreprocessor:
    """Tokeniser and DFG extractor for a single programming language.

    Wraps a Tree-sitter parser and a GraphCodeBERT-compatible
    :class:`RobertaTokenizer` to convert raw source files into the JSONL
    format expected by the training pipeline.  Language-specific grammar rules
    (definition node types, assignment node types, identifier node types, and
    keyword heuristics) are looked up from :data:`LANGUAGE_CONFIG`.

    :param lang_name: One of ``"cpp"``, ``"python"``, ``"java"``, or
        ``"javascript"``.
    :raises ValueError: If ``lang_name`` is not present in
        :data:`LANGUAGE_CONFIG`.
    :raises ImportError: If the Tree-sitter grammar package for the requested
        language is not installed.
    """

    def __init__(self, lang_name: str) -> None:
        self.lang_name = lang_name.lower()
        if self.lang_name not in LANGUAGE_CONFIG:
            raise ValueError(f"Language {lang_name} not supported in registry.")

        self.config = LANGUAGE_CONFIG[self.lang_name]

        try:
            lang_module = __import__(self.config["lib"])
            self.language = Language(lang_module.language())
            self.parser = Parser(self.language)
        except ImportError:
            raise ImportError(f"Please install {self.config['lib']} via pip.")

        self.tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/graphcodebert-base"
        )

    def is_definition(self, node) -> bool:
        """Determine whether a Tree-sitter identifier node is a definition site.

        Checks whether the node's parent is a known definition construct
        (e.g. a variable declaration or function parameter) or the left-hand
        side of an assignment expression, using the grammar rules for the
        current language from :data:`LANGUAGE_CONFIG`.

        :param node: A Tree-sitter node whose ``type`` is an identifier or
            variable.
        :returns: ``True`` if the node is a definition site, ``False``
            otherwise (including when the node has no parent).
        """
        parent = node.parent
        if not parent:
            return False

        if parent.type in self.config["def_nodes"]:
            return True

        if parent.type in self.config["assignment_nodes"]:
            if hasattr(
                parent, "child_by_field_name"
            ) and node == parent.child_by_field_name("left"):
                return True
        return False

    def extract_dfg(self, code_bytes: bytes, tree) -> List[Tuple]:
        """Extract a dataflow graph (DFG) from a parsed Tree-sitter tree.

        Performs two passes over the AST:

        1. **Collection** — records every identifier node and assigns it a
           sequential token position.
        2. **Traversal** — classifies each identifier as a definition or a use
           via :meth:`is_definition`, then emits a directed DFG edge from each
           use site to the most-recent preceding definition of the same name.

        Only use sites that have at least one preceding definition produce an
        edge; forward references and unresolved names are silently ignored.

        :param code_bytes: The UTF-8 encoded source code, used to decode
            variable names from byte offsets.
        :param tree: A Tree-sitter ``Tree`` object produced by parsing
            ``code_bytes``.
        :returns: A list of DFG edges.  Each edge is a 5-tuple
            ``(name, use_pos, "comesFrom", [name], [def_pos])`` where
            ``use_pos`` and ``def_pos`` are integer token-sequence indices.
        """
        root_node = tree.root_node
        var_definitions: Dict[str, List[int]] = defaultdict(list)
        var_uses: Dict[str, List[int]] = defaultdict(list)
        tokens: List = []
        node_to_token_pos: Dict[int, int] = {}

        def collect_identifiers(node) -> None:
            if node.type in self.config["identifier_nodes"]:
                if id(node) not in node_to_token_pos:
                    node_to_token_pos[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                collect_identifiers(child)

        collect_identifiers(root_node)

        def traverse(node) -> None:
            if node.type in self.config["identifier_nodes"]:
                name = code_bytes[node.start_byte : node.end_byte].decode(
                    "utf8", errors="ignore"
                )
                pos = node_to_token_pos.get(id(node), -1)
                if pos != -1:
                    (var_definitions if self.is_definition(node) else var_uses)[
                        name
                    ].append(pos)
            for child in node.children:
                traverse(child)

        traverse(root_node)

        dfg_edges = []
        for name, uses in var_uses.items():
            defs = sorted(var_definitions.get(name, []))
            for use_pos in uses:
                preceding = [d for d in defs if d < use_pos]
                if preceding:
                    dfg_edges.append(
                        (name, use_pos, "comesFrom", [name], [preceding[-1]])
                    )
        return dfg_edges

    def should_keep(self, code: str) -> bool:
        """Apply heuristic quality filters to a raw source string.

        A sample is retained when all three conditions hold:

        * Character length is in the range ``(100, 10000)``.
        * Line count is in the range ``(3, 500)``.
        * At least one language-specific keyword (from
          ``LANGUAGE_CONFIG[lang]["keywords"]``) is present in the source.

        :param code: Raw source code string to evaluate.
        :returns: ``True`` if the sample passes all filters, ``False``
            otherwise.
        """
        if not (100 < len(code) < 10000):
            return False
        if not (3 < code.count("\n") < 500):
            return False
        return any(kw in code for kw in self.config["keywords"])

    def process_sample(self, code: str, idx: int) -> Optional[Dict]:
        """Parse and convert a single source string into a training record.

        Runs the full preprocessing pipeline:

        1. Parses the source with Tree-sitter.
        2. Tokenises with the GraphCodeBERT tokenizer.
        3. Filters out samples with fewer than 10 or more than 450 subword
           tokens.
        4. Extracts the DFG via :meth:`extract_dfg` and filters out samples
           with fewer than 2 edges.
        5. Returns a record dict on success, or ``None`` on any failure
           (including exceptions).

        :param code: Raw source code string to process.
        :param idx: Integer sample index used to construct the unique sample
            ``"idx"`` field in the form ``"<lang>::<idx>"``.
        :returns: A dictionary with keys ``"idx"``, ``"code"``,
            ``"code_tokens"``, ``"dataflow_graph"``, and ``"language"``; or
            ``None`` if the sample is filtered out or an error occurs.
        """
        try:
            code_bytes = code.encode("utf8")
            tree = self.parser.parse(code_bytes)
            tokens = self.tokenizer.tokenize(code, add_prefix_space=True)

            if not (10 < len(tokens) < 450):
                return None

            dfg = self.extract_dfg(code_bytes, tree)
            if not dfg or len(dfg) < 2:
                return None

            return {
                "idx": f"{self.lang_name}::{idx}",
                "code": code,
                "code_tokens": tokens,
                "dataflow_graph": dfg,
                "language": self.lang_name,
            }
        except Exception:
            return None


def stream_dataset(lang: str, output_file: str, max_samples: int) -> None:
    """Stream, filter, and preprocess a language subset from GitHub Code Clean.

    Instantiates a :class:`UniversalPreprocessor` for ``lang``, then iterates
    over the ``"codeparrot/github-code-clean"`` dataset in streaming mode,
    applying :meth:`~UniversalPreprocessor.should_keep` and
    :meth:`~UniversalPreprocessor.process_sample` to each example.  Accepted
    samples are serialised as JSON lines and written to ``output_file``.
    Iteration stops once ``max_samples`` valid samples have been written.

    :param lang: Language identifier passed to :class:`UniversalPreprocessor`;
        must be one of ``"cpp"``, ``"python"``, ``"java"``, or
        ``"javascript"``.
    :param output_file: Path to the output JSONL file.  The file is created or
        overwritten.
    :param max_samples: Maximum number of accepted samples to write.  Streaming
        continues until this count is reached or the dataset is exhausted.
        Pass ``0`` or ``None`` to disable the cap.
    """
    preprocessor = UniversalPreprocessor(lang)
    dataset = load_dataset(
        "codeparrot/github-code-clean",
        preprocessor.config["dataset_name"],
        split="train",
        streaming=True,
    )

    processed_count = 0
    with open(output_file, "w", encoding="utf-8") as f, tqdm(
        desc=f"Processing {lang}"
    ) as pbar:
        for example in dataset:
            if max_samples and processed_count >= max_samples:
                break
            code = example.get("code", "")
            if not preprocessor.should_keep(code):
                continue
            processed = preprocessor.process_sample(code, processed_count)
            if processed:
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                processed_count += 1
                pbar.update(1)


def load_config(config_path: str = "config.json") -> dict:
    """Load the preprocessing section from ``config.json``.

    Returns an empty dict when the file does not exist or contains no
    ``"preprocess"`` key, so callers can safely use ``.get()`` on the result.

    :param config_path: Path to the JSON config file.
    :returns: The ``"preprocess"`` sub-dict, or ``{}`` if absent.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh).get("preprocess", {})


def resolve_langs(lang: str) -> list[str]:
    """Expand the language argument into a list of language identifiers.

    Passing ``"all"`` returns every language present in
    :data:`LANGUAGE_CONFIG`.  Any other value is returned as a single-element
    list after lower-casing.

    :param lang: A language identifier (``"cpp"``, ``"python"``, ``"java"``,
        ``"javascript"``) or the special value ``"all"``.
    :returns: Ordered list of language identifiers to preprocess.
    """
    if lang.lower() == "all":
        return list(LANGUAGE_CONFIG.keys())
    return [lang.lower()]


def main() -> None:
    """Parse CLI arguments, merge config.json defaults, and run preprocessing.

    Resolution order (highest priority first):

    1. CLI flags (``--lang``, ``--max_samples``, ``--config``)
    2. ``config.json`` → ``"preprocess"`` section
    3. Built-in defaults (``lang = "cpp"``, ``max_samples = 10``)

    When ``lang`` is ``"all"``, every language in :data:`LANGUAGE_CONFIG`
    is preprocessed in sequence.  Output files are written to
    ``data/<lang>_processed.jsonl``; the ``data/`` directory is created if it
    does not already exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language to preprocess: cpp | python | java | javascript | all",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of accepted samples to write per language.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        metavar="PATH",
        help="Path to config.json (default: config.json).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Merge: CLI > config.json > built-in default
    lang = args.lang or cfg.get("lang", "cpp")
    max_samples = (
        args.max_samples if args.max_samples is not None else cfg.get("max_samples", 10)
    )

    langs = resolve_langs(lang)
    os.makedirs("data", exist_ok=True)

    print(f"Languages to preprocess : {langs}")
    print(f"Max samples per language: {max_samples}")

    for lg in langs:
        output_path = f"data/{lg}_processed.jsonl"
        print(f"\n--- Preprocessing {lg} → {output_path} ---")
        stream_dataset(lg, output_path, max_samples)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
