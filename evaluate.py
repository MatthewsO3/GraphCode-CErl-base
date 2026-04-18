import json
import os
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
from tqdm import tqdm

SUPPORTED_TS_LANGS = {}
try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjs
    import tree_sitter_python as tspy
    import tree_sitter_cpp as tscpp
    import tree_sitter_erlang as tserl

    SUPPORTED_TS_LANGS = {
        "java": Language(tsjava.language()),
        "javascript": Language(tsjs.language()),
        "python": Language(tspy.language()),
        "cpp": Language(tscpp.language()),
        "erlang": Language(tserl.language()),
    }
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    print(
        "Warning: Some Tree-sitter parsers not found. Source-code evaluation limited to installed parsers."
    )

random.seed(42)
torch.manual_seed(42)


class UnifiedMLMEvaluator:
    """Evaluator for masked language models (MLM) on source code.

    Supports both pre-tokenized JSONL inputs (with optional dataflow graphs)
    and raw source code inputs. Evaluation is performed by masking tokens at a
    configurable ratio and measuring how well the model predicts the originals
    via top-k accuracy and perplexity.

    Dataflow graph (DFG) extraction is performed with Tree-sitter when
    available, covering Java, JavaScript, Python, C++, and Erlang.

    :param model_path: Path to a directory containing a fine-tuned
        ``RobertaForMaskedLM`` checkpoint (compatible with
        ``microsoft/graphcodebert-base`` tokenization).
    :param device: PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        Defaults to CUDA when available, otherwise CPU.
    :param max_seq_length: Maximum total sequence length (code tokens + DFG
        nodes + special tokens).  Inputs longer than this are truncated.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_seq_length: int = 512,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_seq_length = max_seq_length
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/graphcodebert-base"
        )

        print(f"Loading model from {model_path}...")
        self.model = (
            RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        )
        self.parser = Parser() if TS_AVAILABLE else None

    def get_dfg_from_source(self, code: str, lang: str) -> list[tuple]:
        """Extract a dataflow graph (DFG) from raw source code using Tree-sitter.

        Parses the source with the Tree-sitter grammar for the given language,
        identifies identifier nodes that are definitions vs. uses, and emits
        directed edges from each use site to its most-recent preceding
        definition site.

        Returns an empty list when Tree-sitter is unavailable or when ``lang``
        is not in :data:`SUPPORTED_TS_LANGS`.

        :param code: Raw source code to analyse.
        :param lang: Programming language identifier.  Must be one of
            ``"java"``, ``"javascript"``, ``"python"``, ``"cpp"``,
            ``"erlang"``.
        :returns: A list of DFG edges.  Each edge is a 5-tuple of the form
            ``(var_name, use_token_pos, "comesFrom", [var_name], [def_token_pos])``.
        """
        if not TS_AVAILABLE or lang not in SUPPORTED_TS_LANGS:
            return []

        self.parser.set_language(SUPPORTED_TS_LANGS[lang])
        code_bytes = code.encode("utf8")
        tree = self.parser.parse(code_bytes)
        root = tree.root_node

        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node: Any) -> None:
            if node.type in ["identifier", "variable"]:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node: Any) -> bool:
            p = node.parent
            if not p:
                return False

            if lang == "java":
                return p.type in [
                    "local_variable_declaration",
                    "formal_parameter",
                    "method_declaration",
                ] or (
                    p.type == "assignment_expression"
                    and node == p.child_by_field_name("left")
                )
            if lang == "javascript":
                return p.type in [
                    "variable_declarator",
                    "formal_parameters",
                    "function_declaration",
                ] or (
                    p.type == "assignment_expression"
                    and node == p.child_by_field_name("left")
                )
            if lang == "python":
                return p.type in [
                    "assignment",
                    "for_statement",
                    "function_definition",
                    "parameters",
                ] or (
                    p.type == "augmented_assignment"
                    and node == p.child_by_field_name("left")
                )
            if lang == "cpp":
                return p.type in ["declaration", "parameter_declaration"] or (
                    p.type == "assignment_expression"
                    and node == p.child_by_field_name("left")
                )
            if lang == "erlang":
                return p.type in ["variable"] and p.parent.type in [
                    "match_expression",
                    "clause",
                ]
            return False

        def find_vars(node: Any) -> None:
            if node.type in ["identifier", "variable"]:
                name = code_bytes[node.start_byte : node.end_byte].decode(
                    "utf8", "ignore"
                )
                pos = node_map.get(id(node), -1)
                if pos != -1:
                    (defs if is_def(node) else uses)[name].append(pos)
            for child in node.children:
                find_vars(child)

        find_vars(root)
        edges = []
        for name, use_positions in uses.items():
            def_positions = sorted(defs.get(name, []))
            for use_pos in use_positions:
                preds = [d for d in def_positions if d < use_pos]
                if preds:
                    edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))
        return edges

    def build_inputs(
        self,
        masked_tokens: list[str],
        dfg: list,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        """Construct model input tensors from masked token sequence and DFG.

        Combines the masked code token sequence with DFG node placeholders to
        produce the ``input_ids``, 2-D ``attention_mask``, and
        ``position_ids`` tensors expected by GraphCodeBERT.

        The layout of the resulting sequence is::

            [CLS] <code tokens> [SEP] <dfg node tokens> [SEP]

        DFG nodes beyond ``MAX_DFG = min(64, max_length // 4)`` are dropped.
        Code tokens beyond ``MAX_CODE = max_length - MAX_DFG - 3`` are also
        dropped.  The 2-D attention mask allows code tokens to attend to each
        other and DFG nodes to attend to their corresponding code positions and
        to connected DFG siblings.

        :param masked_tokens: Subword tokens with mask positions replaced by
            the tokenizer's ``mask_token``.
        :param dfg: DFG edges as returned by :meth:`get_dfg_from_source` or
            loaded from a pre-processed JSONL sample.
        :param max_length: Maximum total sequence length budget.
        :returns: A dictionary with keys ``"input_ids"``, ``"attention_mask"``,
            and ``"position_ids"``, each a :class:`torch.Tensor` of shape
            ``(1, seq_len)`` (or ``(1, seq_len, seq_len)`` for the attention
            mask).
        """
        MAX_DFG = min(64, max_length // 4)
        MAX_CODE = max_length - MAX_DFG - 3

        masked_tokens = masked_tokens[:MAX_CODE]
        valid_code_len = len(masked_tokens)

        adj, nodes, node_map = defaultdict(list), [], {}
        for edge in dfg:
            var, use_pos = edge[0], edge[1]
            dep_pos_list = edge[4]
            if use_pos >= valid_code_len:
                continue
            if use_pos not in node_map:
                node_map[use_pos] = len(nodes)
                nodes.append((var, use_pos))
            use_idx = node_map[use_pos]
            for def_pos in dep_pos_list:
                if def_pos >= valid_code_len:
                    continue
                if def_pos not in node_map:
                    node_map[def_pos] = len(nodes)
                    nodes.append((var, def_pos))
                adj[use_idx].append(node_map[def_pos])

        nodes = nodes[:MAX_DFG]
        tokens = [self.tokenizer.cls_token] + masked_tokens + [self.tokenizer.sep_token]
        dfg_start = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * len(nodes))
        tokens.append(self.tokenizer.sep_token)

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pos_ids = (
            list(range(valid_code_len + 2)) + [0] * len(nodes) + [valid_code_len + 2]
        )

        mask = np.zeros((len(ids), len(ids)), dtype=bool)
        code_len = valid_code_len + 2
        mask[:code_len, :code_len] = True
        for i in range(len(ids)):
            mask[i, i] = True
        for i, (_, code_pos) in enumerate(nodes):
            dfg_abs, code_abs = dfg_start + i, code_pos + 1
            mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start + i, dfg_start + j
                mask[u, v] = mask[v, u] = True

        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask.tolist()]),
            "position_ids": torch.tensor([pos_ids]),
        }

    def evaluate_sample(
        self,
        sample: dict[str, Any],
        lang: str,
        mask_ratio: float,
        top_k: int,
    ) -> dict[str, Any] | None:
        """Evaluate model MLM performance on a single code sample.

        Accepts either a pre-tokenized sample (containing ``"code_tokens"``
        and optionally ``"dataflow_graph"``) or a raw source sample (containing
        ``"code"`` or ``"source_code"``).

        A random subset of eligible tokens—those whose clean form has more than
        one character after stripping whitespace/continuation markers—is masked
        at the given ``mask_ratio``.  The model predicts the masked tokens and
        the results are recorded as top-1/top-5 hits and log-probabilities.

        :param sample: A single dataset record.  Expected keys:

            * ``"code_tokens"`` *(list[str], optional)* — pre-tokenized subwords.
            * ``"dataflow_graph"`` *(list, optional)* — pre-built DFG edges;
              only used when ``"code_tokens"`` is present.
            * ``"code"`` or ``"source_code"`` *(str)* — raw source used when
              ``"code_tokens"`` is absent.

        :param lang: Programming language identifier passed to
            :meth:`get_dfg_from_source` for raw source samples.
        :param mask_ratio: Fraction of eligible token positions to mask,
            in the range ``(0, 1]``.
        :param top_k: Number of top predictions to retrieve per masked
            position; used when computing the top-k accuracy tensors
            (top-5 is always reported regardless of this value).
        :returns: A dictionary with aggregated results for this sample::

                {
                    "t1":    int,         # number of top-1 hits
                    "t5":    int,         # number of top-5 hits
                    "lp":    list[float], # log-probabilities of true tokens
                    "total": int,         # number of evaluated mask positions
                }

            Returns ``None`` if no token positions could be evaluated (e.g.
            empty input or all tokens are single-character).
        """
        if "code_tokens" in sample:
            code_tokens = sample["code_tokens"]
            dfg = sample.get("dataflow_graph", [])
        else:
            code = sample.get("code") or sample.get("source_code", "")
            code_tokens = self.tokenizer.tokenize(code)
            dfg = self.get_dfg_from_source(code, lang)
        MAX_DFG = min(64, self.max_seq_length // 4)
        MAX_CODE = self.max_seq_length - MAX_DFG - 3

        code_tokens = code_tokens[:MAX_CODE]
        code_tokens = [
            t
            for t in code_tokens
            if t not in (self.tokenizer.cls_token, self.tokenizer.sep_token)
        ]

        candidate_positions = [
            i
            for i, t in enumerate(code_tokens)
            if len(t.replace("Ġ", "").replace("Ċ", "").replace("Â", "")) > 1
        ]

        if not candidate_positions:
            return None
        num_mask = max(1, int(len(candidate_positions) * mask_ratio))
        mask_pos = sorted(
            random.sample(candidate_positions, min(num_mask, len(candidate_positions)))
        )

        orig_ids = []
        for i in mask_pos:
            tid = self.tokenizer.convert_tokens_to_ids(code_tokens[i])
            orig_ids.append(tid if tid != self.tokenizer.unk_token_id else None)

        masked_tokens = code_tokens.copy()
        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        inputs = self.build_inputs(masked_tokens, dfg, self.max_seq_length)

        with torch.no_grad():
            logits = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()}
            ).logits

        results: dict[str, Any] = {"t1": 0, "t5": 0, "lp": [], "total": 0}
        for i, pos in enumerate(mask_pos):
            if orig_ids[i] is None:
                continue

            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            target_id = orig_ids[i]

            _, top_indices = torch.topk(probs, top_k)
            if target_id == top_indices[0]:
                results["t1"] += 1
            if target_id in top_indices[:5]:
                results["t5"] += 1

            results["lp"].append(np.log(max(probs[target_id].item(), 1e-9)))
            results["total"] += 1

        return results if results["total"] > 0 else None


def main() -> None:
    """
    Entry point for the unified MLM evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)
        eval_config = full_config.get("evaluate", {})
        max_length = full_config.get("train", {}).get("max_length", 512)

    model_path = eval_config.get("model")
    langs = eval_config.get("langs", [])
    data_files = eval_config.get("data_files", [])
    mask_ratio = eval_config.get("mask_ratio", 0.15)
    top_k = eval_config.get("top_k", 10)
    max_ex = eval_config.get("max_examples", 1000)

    evaluator = UnifiedMLMEvaluator(model_path, max_seq_length=max_length)

    all_metrics = []
    for file_path, lang in zip(data_files, langs):
        print(f"\n--- Evaluating {lang} from {file_path} ---")
        metrics: dict[str, Any] = {"t1": 0, "t5": 0, "total": 0, "lp": []}

        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            lines = f.readlines()
            if max_ex:
                lines = lines[:max_ex]

            for line in tqdm(lines, desc=f"Processing {lang}"):
                sample = json.loads(line)
                res = evaluator.evaluate_sample(sample, lang.lower(), mask_ratio, top_k)
                if res:
                    metrics["t1"] += res["t1"]
                    metrics["t5"] += res["t5"]
                    metrics["total"] += res["total"]
                    metrics["lp"].extend(res["lp"])

        if metrics["total"] > 0:
            ppl = np.exp(-np.mean(metrics["lp"]))
            print(
                f"Results for {lang}: Top-1: {metrics['t1']/metrics['total']:.2%}, PPL: {ppl:.4f}"
            )
            all_metrics.append(metrics)


if __name__ == "__main__":
    main()
