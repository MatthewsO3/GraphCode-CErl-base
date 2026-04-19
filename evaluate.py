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
        n_nodes = len(nodes)
        adj = {
            i: [j for j in neighbors if j < n_nodes]
            for i, neighbors in adj.items()
            if i < n_nodes
        }
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

        # Store original token strings alongside IDs so accuracy uses string
        # comparison (matching the reference script) while perplexity uses the
        # stored ID directly — never a re-lookup that could hit unk_token_id.
        orig_toks = [code_tokens[i] for i in mask_pos]

        top1_correct = top5_correct = 0
        log_probs: list[float] = []

        for i, pos in enumerate(mask_pos):
            if orig_ids[i] is None:
                continue

            probs = torch.softmax(logits[0, pos + 1], dim=-1)

            # Accuracy: compare predicted token strings against the original string.
            _, top_indices = torch.topk(probs, top_k)
            top_preds = self.tokenizer.convert_ids_to_tokens(top_indices)
            for rank, pred in enumerate(top_preds, 1):
                if pred == orig_toks[i]:
                    if rank == 1:
                        top1_correct += 1
                    if rank <= 5:
                        top5_correct += 1
                    break

            # Perplexity: use the stored ID directly, never re-lookup.
            correct_prob = probs[orig_ids[i]].item()
            log_probs.append(np.log(max(correct_prob, 1e-9)))

        if not log_probs:
            return None

        return {
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "num_masked":   len(log_probs),
            "log_probs":    log_probs,
        }


def print_lang_results(lang: str, metrics: dict[str, Any]) -> None:
    """Print a formatted results block for one language, matching the reference
    script's output style.

    :param lang: Display label for the language.
    :param metrics: Aggregated metrics dict as built in :func:`main`.
    """
    total = metrics["total_masked_tokens"]
    print("\n" + "=" * 70)
    print(f"Evaluation Results — {lang}".center(70))
    print("=" * 70)
    print(f"Snippets evaluated:     {metrics['snippets_evaluated']}")
    print(f"Total masked tokens:    {total}")
    print("-" * 70)
    print(f"Top-1 Accuracy:         {metrics['top1_accuracy']:.2%}"
          f" ({metrics['top1_correct']}/{total})")
    print(f"Top-5 Accuracy:         {metrics['top5_accuracy']:.2%}"
          f" ({metrics['top5_correct']}/{total})")
    print(f"Perplexity:             {metrics['perplexity']:.4f}")
    print("=" * 70 + "\n")


def save_lang_results(
    lang: str,
    metrics: dict[str, Any],
    snippets_skipped: int,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Print and persist per-language evaluation results.

    Prints a formatted results block, then writes a single JSON file to
    ``<output_dir>/eval_<lang>.json`` with the same structure produced by
    the reference ``python_eval.py`` script.

    :param lang: Language identifier used in the filename.
    :param metrics: Aggregated metrics dict with keys ``"snippets_evaluated"``,
        ``"total_masked_tokens"``, ``"top1_correct"``, ``"top5_correct"``,
        ``"top1_accuracy"``, ``"top5_accuracy"``, and ``"perplexity"``.
    :param snippets_skipped: Number of samples that returned no results.
    :param config: Configuration dict to embed in the saved file.
    :param output_dir: Directory to write the JSON file into.
    """
    if metrics["total_masked_tokens"] == 0:
        print(f"[eval] No results for {lang} — skipping save.")
        return

    print_lang_results(lang, metrics)

    results_serializable = {
        "snippets_evaluated":  metrics["snippets_evaluated"],
        "total_masked_tokens": metrics["total_masked_tokens"],
        "top1_accuracy":       float(metrics["top1_accuracy"]),
        "top5_accuracy":       float(metrics["top5_accuracy"]),
        "perplexity":          float(metrics["perplexity"]),
        "top1_correct":        int(metrics["top1_correct"]),
        "top5_correct":        int(metrics["top5_correct"]),
        "snippets_skipped":    snippets_skipped,
        "config":              config,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_{lang}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results_serializable, fh, indent=2)

    print(f"[eval] Saved {lang} results → {out_path}")


def main() -> None:
    """Entry point for the unified MLM evaluation script.

    Reads all configuration from ``config.json``.  The model path defaults to
    ``<train.output_dir>/best_model`` so evaluation always runs against the
    best checkpoint produced by training, but can be overridden by setting
    ``evaluate.model`` explicitly.

    Per-language results are saved to
    ``<train.output_dir>/best_model/eval_<lang>.json``.  A combined summary
    across all languages is saved to ``eval_summary.json`` in the same
    directory.  Both files use the same indented JSON format as the reference
    ``python_eval.py`` script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)

    eval_config  = full_config.get("evaluate", {})
    train_config = full_config.get("train", {})
    max_length   = train_config.get("max_length", 512)

    # Resolve the model path: explicit evaluate.model > best_model under output_dir.
    output_dir = train_config.get("output_dir")
    if not output_dir:
        raise ValueError("'output_dir' must be set under 'train' in config.json.")

    best_model_dir = Path(output_dir) / "best_model"
    model_path = eval_config.get("model") or str(best_model_dir)

    langs      = eval_config.get("langs", [])
    data_files = eval_config.get("data_files", [])
    mask_ratio = eval_config.get("mask_ratio", 0.15)
    top_k      = eval_config.get("top_k", 10)
    max_ex     = eval_config.get("max_examples", 1000)

    print(f"\n{'=' * 70}")
    print(f"  MLM EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Model      : {model_path}")
    print(f"  Languages  : {langs}")
    print(f"  Mask ratio : {mask_ratio}")
    print(f"  Max samples: {max_ex}")
    print(f"  Output dir : {best_model_dir}")
    print(f"{'=' * 70}\n")

    evaluator = UnifiedMLMEvaluator(model_path, max_seq_length=max_length)

    # lang_metrics stores the full aggregated dict per language so we can
    # compute a weighted combined perplexity at the end (same as reference script).
    lang_metrics: dict[str, dict[str, Any]] = {}

    for file_path, lang in zip(data_files, langs):
        print(f"\n--- Evaluating {lang} from {file_path} ---")

        total_top1 = total_top5 = total_masked = snippets = snippets_skipped = 0
        all_log_probs: list[float] = []

        if not os.path.exists(file_path):
            print(f"[eval] Warning: {file_path} not found — skipping {lang}.")
            continue

        with open(file_path, "r") as f:
            lines = f.readlines()
            if max_ex:
                lines = lines[:max_ex]

        for line in tqdm(lines, desc=f"Processing {lang}"):
            sample = json.loads(line)
            res = evaluator.evaluate_sample(sample, lang.lower(), mask_ratio, top_k)
            if not res:
                snippets_skipped += 1
                continue

            total_top1    += res["top1_correct"]
            total_top5    += res["top5_correct"]
            total_masked  += res["num_masked"]
            all_log_probs += res["log_probs"]
            snippets      += 1

        if total_masked == 0:
            print(f"[eval] No valid results for {lang}.")
            continue

        mean_lp = float(np.mean(all_log_probs))
        metrics: dict[str, Any] = {
            "snippets_evaluated":  snippets,
            "total_masked_tokens": total_masked,
            "top1_correct":        total_top1,
            "top5_correct":        total_top5,
            "top1_accuracy":       float(total_top1 / total_masked),
            "top5_accuracy":       float(total_top5 / total_masked),
            "perplexity":          float(np.exp(-mean_lp)),
            "_mean_log_prob":      mean_lp,   # kept for weighted combined PPL
            "_snippets_skipped":   snippets_skipped,
        }

        lang_config = {
            "language":     lang,
            "data_file":    file_path,
            "mask_ratio":   mask_ratio,
            "top_k":        top_k,
            "max_examples": max_ex,
            "model":        model_path,
            "max_length":   max_length,
        }
        save_lang_results(lang, metrics, snippets_skipped, lang_config, best_model_dir)
        lang_metrics[lang] = metrics

    # Combined metrics across all evaluated languages (weighted perplexity).
    if lang_metrics:
        all_top1   = sum(v["top1_correct"]        for v in lang_metrics.values())
        all_top5   = sum(v["top5_correct"]        for v in lang_metrics.values())
        all_masked = sum(v["total_masked_tokens"] for v in lang_metrics.values())
        all_snips  = sum(v["snippets_evaluated"]  for v in lang_metrics.values())

        weighted_lp = sum(
            v["_mean_log_prob"] * v["total_masked_tokens"]
            for v in lang_metrics.values()
        ) / all_masked

        combined: dict[str, Any] = {
            "snippets_evaluated":  all_snips,
            "total_masked_tokens": all_masked,
            "top1_correct":        all_top1,
            "top5_correct":        all_top5,
            "top1_accuracy":       float(all_top1 / all_masked),
            "top5_accuracy":       float(all_top5 / all_masked),
            "perplexity":          float(np.exp(-weighted_lp)),
        }
        print_lang_results("Combined", combined)

        # Save combined summary as a single JSON file matching the reference format.
        combined_skipped = sum(v["_snippets_skipped"] for v in lang_metrics.values())
        combined_config = {
            "langs":        langs,
            "data_files":   data_files,
            "mask_ratio":   mask_ratio,
            "top_k":        top_k,
            "max_examples": max_ex,
            "model":        model_path,
            "max_length":   max_length,
        }
        combined_serializable = {
            "snippets_evaluated":  combined["snippets_evaluated"],
            "total_masked_tokens": combined["total_masked_tokens"],
            "top1_accuracy":       float(combined["top1_accuracy"]),
            "top5_accuracy":       float(combined["top5_accuracy"]),
            "perplexity":          float(combined["perplexity"]),
            "top1_correct":        int(combined["top1_correct"]),
            "top5_correct":        int(combined["top5_correct"]),
            "snippets_skipped":    combined_skipped,
            "config":              combined_config,
        }
        summary_path = best_model_dir / "eval_summary.json"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(combined_serializable, fh, indent=2)
        print(f"[eval] Combined summary saved → {summary_path}")


if __name__ == "__main__":
    main()