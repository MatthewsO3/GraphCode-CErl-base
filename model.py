import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Locate the project root directory by searching for ``config.json``.

    Walks upward from ``start_path`` (or the directory containing this file)
    until a directory containing ``config.json`` is found.

    :param start_path: Directory from which to begin the upward search.
        Defaults to the parent directory of this source file.
    :returns: The first ancestor directory that contains ``config.json``.
    """
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current: Path = start_path
    while True:
        config_path: Path = current / "config.json"
        if config_path.exists():
            return current

        parent: Path = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find project root. "
                "Make sure config.json exists in the project root directory."
            )
        current = parent


def load_config() -> Dict:
    """Load and return the project configuration from ``config.json``.

    Resolves the project root via :func:`find_project_root` and reads the JSON
    configuration file located there.

    :returns: The parsed contents of ``config.json`` as a dictionary.
    """
    project_root: Path = find_project_root()
    config_path: Path = project_root / "config.json"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


class GraphCodeBERTDataset(Dataset):
    """PyTorch ``Dataset`` for GraphCodeBERT pre-training on code + DFG data.

    Reads a JSONL file where each line is a JSON object containing at minimum:

    * ``"code_tokens"`` *(list[str])* — subword tokens for the code snippet.
    * ``"dataflow_graph"`` *(list, optional)* — DFG edges in the form
      ``(var, use_pos, rel, [var], [def_pos])``.

    Each sample is converted to padded tensors with a 2-D attention mask that
    encodes code-to-code, code-to-DFG-node, and DFG-node-to-DFG-node
    attention permissions.

    :param jsonl_file: Path to the JSONL data file.
    :param tokenizer: A Hugging Face tokenizer compatible with
        ``microsoft/graphcodebert-base`` (e.g. :class:`RobertaTokenizer`).
    :param max_length: Maximum padded sequence length (code tokens + DFG
        placeholder tokens + 3 special tokens).  When ``None``, the value is
        read from ``config.json`` under ``train.max_length``; falls back to
        ``512`` if the config cannot be found.
    """

    def __init__(
        self,
        jsonl_file: str,
        tokenizer: RobertaTokenizer,
        max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer: RobertaTokenizer = tokenizer
        self.max_length: int

        if max_length is not None:
            self.max_length = max_length
        else:
            try:
                config: Dict = load_config()
                self.max_length = config.get("train", {}).get("max_length", 512)
            except FileNotFoundError:
                self.max_length = 512

        self.samples: List[Dict] = []
        print(f"Loading and processing data from {jsonl_file}...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        :returns: Number of loaded samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the feature dictionary for the sample at ``idx``.

        Delegates to :meth:`convert_sample_to_features`.

        :param idx: Index into the list of loaded samples.
        :returns: Feature dictionary as produced by
            :meth:`convert_sample_to_features`.
        """
        return self.convert_sample_to_features(self.samples[idx])

    def convert_sample_to_features(self, sample: Dict) -> Dict:
        """Convert a raw JSONL sample to padded model-ready tensors.

        Constructs the GraphCodeBERT sequence layout::

            [CLS] <code_tokens> [SEP] <dfg_node_placeholders> [SEP] [PAD ...]

        and builds the corresponding 2-D boolean attention mask, respecting:

        * Full bidirectional attention within the code segment.
        * Bidirectional attention between each DFG node and its corresponding
          code token.
        * Bidirectional attention between DFG nodes connected by an edge.
        * Self-attention for every position.

        DFG nodes and code tokens that exceed their respective budget
        (``MAX_DFG`` and ``MAX_CODE``) are silently dropped.  An
        :class:`AssertionError` is raised if the budgets are miscalculated and
        the combined length still exceeds ``self.max_length``.

        :param sample: Raw sample dictionary with keys:

            * ``"code_tokens"`` *(list[str])* — pre-tokenized subwords.
            * ``"dataflow_graph"`` *(list, optional)* — DFG edge tuples.

        :returns: A dictionary with the following keys:

            * ``"input_ids"`` (:class:`torch.Tensor`, shape ``[max_length]``,
              dtype ``long``) — token ids, padded.
            * ``"attention_mask"`` (:class:`torch.Tensor`, shape
              ``[max_length, max_length]``, dtype ``bool``) — 2-D attention
              mask.
            * ``"position_idx"`` (:class:`torch.Tensor`, shape ``[max_length]``,
              dtype ``long``) — position indices; DFG nodes use position 0.
            * ``"dfg_info"`` *(dict)* — raw DFG metadata:

              - ``"nodes"`` *(list[tuple[str, int]])* — retained DFG nodes as
                ``(variable_name, code_token_position)`` pairs.
              - ``"edges"`` *(list[tuple[int, int]])* — directed edges between
                DFG node indices ``(use_idx, def_idx)``.
        """
        code_tokens: List[str] = sample["code_tokens"]
        dfg: List[Tuple] = sample.get("dataflow_graph", [])

        MAX_DFG: int = min(64, self.max_length // 4)
        MAX_CODE: int = self.max_length - MAX_DFG - 3

        code_tokens = code_tokens[:MAX_CODE]
        valid_code_len: int = len(code_tokens)

        adj: Dict[int, List[int]] = defaultdict(list)
        dfg_nodes: List[Tuple[str, int]] = []
        node_to_idx: Dict[int, int] = {}

        for var, use_pos, _, _, dep_pos_list in dfg:
            if use_pos >= valid_code_len:
                continue
            if use_pos not in node_to_idx:
                node_to_idx[use_pos] = len(dfg_nodes)
                dfg_nodes.append((var, use_pos))
            use_idx: int = node_to_idx[use_pos]

            for def_pos in dep_pos_list:
                if def_pos >= valid_code_len:
                    continue
                if def_pos not in node_to_idx:
                    node_to_idx[def_pos] = len(dfg_nodes)
                    dfg_nodes.append((var, def_pos))
                adj[use_idx].append(node_to_idx[def_pos])

        if len(dfg_nodes) > MAX_DFG:
            keep: Set[int] = set(range(MAX_DFG))
            dfg_nodes = dfg_nodes[:MAX_DFG]
            adj = defaultdict(
                list,
                {
                    i: [j for j in adjs if j in keep]
                    for i, adjs in adj.items()
                    if i in keep
                },
            )

        dfg_token_count: int = len(dfg_nodes)

        total: int = valid_code_len + dfg_token_count + 3
        assert (
            total <= self.max_length
        ), f"Still too long after capping: code={valid_code_len} dfg={dfg_token_count} total={total}"

        tokens: List[str] = (
            [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        )
        dfg_start_pos: int = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * dfg_token_count)
        tokens.append(self.tokenizer.sep_token)

        input_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)
        position_idx: List[int] = (
            list(range(valid_code_len + 2))
            + [0] * dfg_token_count
            + [valid_code_len + 2]
        )

        padding_len: int = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        position_idx += [0] * padding_len

        attn_mask: np.ndarray = np.zeros(
            (self.max_length, self.max_length), dtype=np.bool_
        )
        code_len: int = valid_code_len + 2

        attn_mask[:code_len, :code_len] = True

        for i in range(len(tokens)):
            attn_mask[i, i] = True

        for i, (_, code_pos) in enumerate(dfg_nodes):
            dfg_abs: int = dfg_start_pos + i
            code_abs: int = code_pos + 1
            attn_mask[dfg_abs, code_abs] = True
            attn_mask[code_abs, dfg_abs] = True

        for i, adjs in adj.items():
            for j in adjs:
                u: int = dfg_start_pos + i
                v: int = dfg_start_pos + j
                attn_mask[u, v] = True
                attn_mask[v, u] = True

        assert len(input_ids) == self.max_length
        assert len(position_idx) == self.max_length
        assert attn_mask.shape == (self.max_length, self.max_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "position_idx": torch.tensor(position_idx, dtype=torch.long),
            "dfg_info": {
                "nodes": dfg_nodes,
                "edges": [(i, j) for i, adjs in adj.items() for j in adjs],
            },
        }


class GraphCodeBERTWithEdgePrediction(nn.Module):
    """GraphCodeBERT model with an auxiliary DFG edge-prediction head.

    Wraps a ``RobertaForMaskedLM`` backbone with a two-layer MLP edge
    classifier.  The model accepts two input formats produced by different
    language-specific data pipelines:

    **Erlang format**

    * ``position_idx`` — ``[batch, seq_len]``
    * ``edge_candidates`` — ``[batch, max_edges, 2]`` where each row is
      ``[node1_pos, node2_pos]``
    * ``edge_labels`` — ``[batch, max_edges]``
    * ``alignment_candidates``, ``alignment_labels`` (optional, unused)
    * ``dfg_start_idx`` (optional, unused)

    **C++ format**

    * ``position_ids`` — ``[batch, seq_len]``
    * ``edge_batch_idx``, ``edge_node1_pos``, ``edge_node2_pos`` — 1-D tensors
      of equal length enumerating all candidate edge triplets across the batch
    * ``edge_labels`` — ``[num_edges]``

    When Erlang-format edges are supplied, they are converted to C++ format
    internally via :meth:`_convert_erlang_edges_to_cpp` before the edge head
    is applied.

    The combined training loss is ``mlm_loss + edge_loss`` when both targets
    are available; otherwise whichever non-``None`` component is present is
    returned as the total loss.

    :param base_model_name: Hugging Face model identifier or local path for
        the ``RobertaForMaskedLM`` backbone.  When ``None``, the value is read
        from ``config.json`` under ``model.base_model``; falls back to
        ``"microsoft/graphcodebert-base"``.
    """

    def __init__(self, base_model_name: Optional[str] = None) -> None:
        super().__init__()

        if base_model_name is None:
            try:
                config: Dict = load_config()
                base_model_name = config.get("model", {}).get(
                    "base_model", "microsoft/graphcodebert-base"
                )
            except FileNotFoundError:
                base_model_name = "microsoft/graphcodebert-base"

        self.roberta_mlm: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained(
            base_model_name
        )
        hidden_size: int = self.roberta_mlm.config.hidden_size

        try:
            config = load_config()
            hidden_dropout: float = config.get("model", {}).get(
                "hidden_dropout_prob", 0.2
            )
            attention_dropout: float = config.get("model", {}).get(
                "attention_probs_dropout_prob", 0.2
            )
        except FileNotFoundError:
            hidden_dropout = 0.2
            attention_dropout = 0.2

        self.roberta_mlm.config.hidden_dropout_prob = hidden_dropout
        self.roberta_mlm.config.attention_probs_dropout_prob = attention_dropout

        self.edge_classifier: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        # C++ format
        position_ids: Optional[torch.Tensor] = None,
        edge_batch_idx: Optional[torch.Tensor] = None,
        edge_node1_pos: Optional[torch.Tensor] = None,
        edge_node2_pos: Optional[torch.Tensor] = None,
        # Erlang format
        position_idx: Optional[torch.Tensor] = None,
        edge_candidates: Optional[torch.Tensor] = None,
        alignment_candidates: Optional[torch.Tensor] = None,
        alignment_labels: Optional[torch.Tensor] = None,
        dfg_start_idx: Optional[torch.Tensor] = None,
        edge_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run the combined MLM and edge-prediction forward pass.

        Accepts either Erlang- or C++-format edge inputs and normalises them
        to a common representation before computing losses.

        Position IDs are resolved in the following priority order:

        1. ``position_ids`` (C++ format, explicit).
        2. ``position_idx`` (Erlang format, aliased to ``position_ids``).
        3. A fallback sequence ``[0, 1, ..., seq_len-1]`` broadcast over the
           batch.

        Erlang-format ``edge_candidates`` are converted to C++-format flat
        tensors via :meth:`_convert_erlang_edges_to_cpp` when
        ``edge_batch_idx`` is not already supplied.

        :param input_ids: Token ids, shape ``[batch, seq_len]``.
        :param attention_mask: 2-D boolean attention mask, shape
            ``[batch, seq_len, seq_len]``.
        :param labels: MLM target ids with ``-100`` at non-masked positions,
            shape ``[batch, seq_len]``. When ``None``, MLM loss is not
            computed.
        :param position_ids: Position indices in C++ format,
            shape ``[batch, seq_len]``.
        :param edge_batch_idx: Batch index for each candidate edge (C++
            format), shape ``[num_edges]``.
        :param edge_node1_pos: Sequence position of the first node in each
            candidate edge (C++ format), shape ``[num_edges]``.
        :param edge_node2_pos: Sequence position of the second node in each
            candidate edge (C++ format), shape ``[num_edges]``.
        :param position_idx: Position indices in Erlang format (alias for
            ``position_ids``), shape ``[batch, seq_len]``.
        :param edge_candidates: Edge pairs in Erlang format,
            shape ``[batch, max_edges, 2]``.
        :param alignment_candidates: Unused alignment data (reserved for
            future use).
        :param alignment_labels: Unused alignment labels (reserved for future
            use).
        :param dfg_start_idx: Unused DFG start index (reserved for future
            use).
        :param edge_labels: Binary edge labels, shape ``[num_edges]``.
        :returns: A dictionary with three keys:

            * ``"loss"`` (:class:`torch.Tensor`) — combined scalar loss
              (``mlm_loss + edge_loss``), or whichever component is available.
              A zero-gradient tensor is returned when neither loss is computable.
            * ``"mlm_loss"`` (:class:`torch.Tensor` or ``None``) — masked
              language modelling cross-entropy loss.
            * ``"edge_loss"`` (:class:`torch.Tensor` or ``None``) — binary
              cross-entropy loss over the edge classifier logits.
        """
        if position_ids is None and position_idx is not None:
            position_ids = position_idx
        elif position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if edge_candidates is not None and edge_batch_idx is None:
            edge_batch_idx, edge_node1_pos, edge_node2_pos = (
                self._convert_erlang_edges_to_cpp(edge_candidates)
            )

        mlm_outputs = self.roberta_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        mlm_loss: Optional[torch.Tensor] = (
            mlm_outputs.loss if labels is not None else None
        )

        edge_loss: Optional[torch.Tensor] = None
        if (
            edge_batch_idx is not None
            and len(edge_batch_idx) > 0
            and edge_node1_pos is not None
            and edge_node2_pos is not None
            and edge_labels is not None
        ):
            hidden_states: torch.Tensor = mlm_outputs.hidden_states[-1]

            node1_repr: torch.Tensor = hidden_states[edge_batch_idx, edge_node1_pos]
            node2_repr: torch.Tensor = hidden_states[edge_batch_idx, edge_node2_pos]
            edge_repr: torch.Tensor = torch.cat([node1_repr, node2_repr], dim=-1)
            edge_logits: torch.Tensor = self.edge_classifier(edge_repr).squeeze(-1)

            edge_labels_float: torch.Tensor = (
                edge_labels.float()
                if edge_labels.dtype != torch.float32
                else edge_labels
            )
            edge_loss = nn.functional.binary_cross_entropy_with_logits(
                edge_logits, edge_labels_float
            )

        total_loss: torch.Tensor
        if mlm_loss is not None and edge_loss is not None:
            total_loss = mlm_loss + edge_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif edge_loss is not None:
            total_loss = edge_loss
        else:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "edge_loss": edge_loss,
        }

    def _convert_erlang_edges_to_cpp(
        self, edge_candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert Erlang-format edge candidates to flat C++-format tensors.

        Erlang format stores edges as a padded 3-D tensor where zero-valued
        rows ``[0, 0]`` are treated as padding and ignored.

        :param edge_candidates: Padded edge pairs, shape
            ``[batch_size, max_edges, 2]``, where ``edge_candidates[b, e]``
            is ``[node1_pos, node2_pos]`` for edge *e* in batch item *b*.
            Rows where both positions are ``0`` are considered padding.
        :returns: A 3-tuple ``(edge_batch_idx, edge_node1_pos, edge_node2_pos)``
            of 1-D ``long`` tensors on the same device as ``edge_candidates``.
            Each element at index *i* describes one valid edge:

            * ``edge_batch_idx[i]`` — batch item index.
            * ``edge_node1_pos[i]`` — sequence position of node 1.
            * ``edge_node2_pos[i]`` — sequence position of node 2.

            All three tensors are empty (length 0) when no valid edges are
            found.
        """
        batch_size: int = edge_candidates.shape[0]

        batch_indices: List[int] = []
        node1_positions: List[int] = []
        node2_positions: List[int] = []

        for batch_idx in range(batch_size):
            edges: torch.Tensor = edge_candidates[batch_idx]
            valid_edges: torch.Tensor = edges[(edges[:, 0] != 0) | (edges[:, 1] != 0)]

            if len(valid_edges) > 0:
                batch_indices.extend([batch_idx] * len(valid_edges))
                node1_positions.extend(valid_edges[:, 0].tolist())
                node2_positions.extend(valid_edges[:, 1].tolist())

        device: torch.device = edge_candidates.device

        if len(batch_indices) > 0:
            return (
                torch.tensor(batch_indices, dtype=torch.long, device=device),
                torch.tensor(node1_positions, dtype=torch.long, device=device),
                torch.tensor(node2_positions, dtype=torch.long, device=device),
            )
        else:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
            )

    def save_pretrained(self, save_directory: str) -> None:
        """Save the backbone ``roberta_mlm`` weights to ``save_directory``.

        Only the underlying ``RobertaForMaskedLM`` weights are persisted so
        that the checkpoint is compatible with
        ``RobertaForMaskedLM.from_pretrained()`` and can be loaded by
        language-specific fine-tuning scripts without this wrapper class.

        .. note::
            The ``edge_classifier`` head weights are **not** saved.  The head
            will be re-initialised randomly on the next call to
            :meth:`from_pretrained`.

        :param save_directory: Directory path where the model weights and
            config will be written.
        """
        self.roberta_mlm.save_pretrained(save_directory)
        print(f"✓ Base model saved to {save_directory}")
        print(f"  Note: edge_classifier is NOT saved (it gets re-initialized on load)")

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "GraphCodeBERTWithEdgePrediction":
        """Instantiate the model from a pretrained backbone checkpoint.

        Creates a new :class:`GraphCodeBERTWithEdgePrediction` instance using
        ``model_name_or_path`` as the backbone.  The ``edge_classifier`` head
        receives a fresh random initialisation.

        :param model_name_or_path: Hugging Face model identifier or local
            directory path accepted by ``RobertaForMaskedLM.from_pretrained()``.
        :param kwargs: Additional keyword arguments (accepted for API
            compatibility but currently unused).
        :returns: A newly constructed model instance with the backbone loaded
            from ``model_name_or_path``.
        """
        instance: "GraphCodeBERTWithEdgePrediction" = cls(model_name_or_path)
        return instance


@dataclass
class MLMWithEdgePredictionCollator:
    """Data collator that applies dynamic MLM masking and samples DFG edge pairs.

    Combines multiple :class:`GraphCodeBERTDataset` samples into a single
    batch, applies token masking for the MLM objective, and samples positive
    and negative DFG edge candidate pairs for the edge-prediction auxiliary
    task.

    Token masking follows the standard BERT strategy:

    * 80 % of selected positions → ``[MASK]``
    * 10 % of selected positions → random vocabulary token
    * 10 % of selected positions → unchanged (original token)

    Only tokens with a ``position_idx > 1`` (i.e. non-special code tokens,
    excluding the final ``[SEP]``) are eligible for masking.

    Edge candidates are sampled uniformly from all unique unordered pairs of
    DFG nodes, with the sample size capped at ``max_pairs = 20`` per example.
    Positive labels (``1``) are assigned to pairs that appear in the DFG edge
    set (in either direction); negative labels (``0``) to all other pairs.

    :param tokenizer: Tokenizer used to obtain ``pad_token_id``,
        ``mask_token_id``, and ``vocab_size``.
    :param mlm_probability: Fraction of eligible code tokens to mask per
        sample.  Loaded from ``config.json`` under ``train.mlm_probability``
        when ``None``; falls back to ``0.15``.
    :param edge_sample_ratio: Fraction of all possible node pairs to sample
        as edge candidates.  Loaded from ``config.json`` under
        ``model.edge_sample_ratio`` when ``None``; falls back to ``0.3``.
    """

    tokenizer: RobertaTokenizer
    mlm_probability: Optional[float] = None
    edge_sample_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        """Resolve ``None`` hyperparameters from ``config.json`` or defaults."""
        if self.mlm_probability is None or self.edge_sample_ratio is None:
            try:
                config: Dict = load_config()
                train_config: Dict = config.get("train", {})
                model_config: Dict = config.get("model", {})

                if self.mlm_probability is None:
                    self.mlm_probability = train_config.get("mlm_probability", 0.15)
                if self.edge_sample_ratio is None:
                    self.edge_sample_ratio = model_config.get("edge_sample_ratio", 0.3)
            except FileNotFoundError:
                if self.mlm_probability is None:
                    self.mlm_probability = 0.15
                if self.edge_sample_ratio is None:
                    self.edge_sample_ratio = 0.3

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a list of dataset samples into a padded batch.

        Applies dynamic MLM masking and samples DFG edge candidates for every
        example before stacking into batch tensors.

        :param examples: A list of feature dictionaries as returned by
            :meth:`GraphCodeBERTDataset.convert_sample_to_features`.  All
            examples must share the same ``max_length`` (i.e. the same
            ``input_ids`` length).
        :returns: A batch dictionary with the following keys:

            * ``"input_ids"`` (:class:`torch.Tensor`, ``[batch, seq_len]``,
              ``long``) — token IDs with masked positions replaced.
            * ``"attention_mask"`` (:class:`torch.Tensor`,
              ``[batch, seq_len, seq_len]``, ``bool``) — 2-D attention masks.
            * ``"position_ids"`` (:class:`torch.Tensor`, ``[batch, seq_len]``,
              ``long``) — position indices.
            * ``"labels"`` (:class:`torch.Tensor`, ``[batch, seq_len]``,
              ``long``) — MLM targets; ``-100`` at non-masked positions and at
              padding positions.
            * ``"edge_batch_idx"`` (:class:`torch.Tensor`, ``[num_edges]``,
              ``long``) — batch index for each sampled edge candidate.
            * ``"edge_node1_pos"`` (:class:`torch.Tensor`, ``[num_edges]``,
              ``long``) — sequence position of edge endpoint 1.
            * ``"edge_node2_pos"`` (:class:`torch.Tensor`, ``[num_edges]``,
              ``long``) — sequence position of edge endpoint 2.
            * ``"edge_labels"`` (:class:`torch.Tensor`, ``[num_edges]``,
              ``float``) — ``1.0`` if the pair is a true DFG edge, ``0.0``
              otherwise.

            When no valid edge candidates can be sampled, the four edge tensors
            are empty (length 0).
        """
        batch_size: int = len(examples)
        max_seq_length: int = examples[0]["input_ids"].shape[0]

        input_ids: torch.Tensor = torch.stack([ex["input_ids"] for ex in examples])
        attn_mask: torch.Tensor = torch.stack([ex["attention_mask"] for ex in examples])
        pos_idx: torch.Tensor = torch.stack([ex["position_idx"] for ex in examples])

        assert input_ids.shape == (batch_size, max_seq_length)
        assert attn_mask.shape == (batch_size, max_seq_length, max_seq_length)
        assert pos_idx.shape == (batch_size, max_seq_length)

        labels: torch.Tensor = input_ids.clone()
        masked_ids: torch.Tensor = input_ids.clone()

        for i in range(batch_size):
            code_indices: torch.Tensor = (pos_idx[i] > 1).nonzero(as_tuple=True)[0]
            if len(code_indices) > 1:
                code_indices = code_indices[:-1]
            if len(code_indices) == 0:
                continue
            num_mask: int = max(1, int(len(code_indices) * self.mlm_probability))
            mask_pos: torch.Tensor = code_indices[
                torch.randperm(len(code_indices))[:num_mask]
            ]
            for pos in mask_pos:
                if random.random() < 0.8:
                    masked_ids[i, pos] = self.tokenizer.mask_token_id
                elif random.random() < 0.5:
                    masked_ids[i, pos] = random.randint(
                        0, self.tokenizer.vocab_size - 1
                    )
            mask_ind: torch.Tensor = torch.zeros_like(labels[i], dtype=torch.bool)
            mask_ind[mask_pos] = True
            labels[i, ~mask_ind] = -100
        labels[masked_ids == self.tokenizer.pad_token_id] = -100

        edge_pairs: List[Tuple[int, int, int, int]] = []
        max_pairs: int = 20
        for i in range(batch_size):
            if "dfg_info" not in examples[i]:
                continue
            dfg_nodes: List[Tuple[str, int]] = examples[i]["dfg_info"]["nodes"]
            dfg_edges: List[Tuple[int, int]] = examples[i]["dfg_info"]["edges"]
            if len(dfg_nodes) < 2:
                continue

            edge_set: Set[Tuple[int, int]] = set(dfg_edges)
            edge_set.update((v, u) for u, v in dfg_edges)

            num_nodes: int = len(dfg_nodes)
            num_pairs: int = min(
                max_pairs,
                int(num_nodes * (num_nodes - 1) / 2 * self.edge_sample_ratio),
            )
            sampled: Set[Tuple[int, int]] = set()
            attempts: int = 0
            while len(sampled) < num_pairs and attempts < num_pairs * 3:
                u: int = random.randint(0, num_nodes - 1)
                v: int = random.randint(0, num_nodes - 1)
                if u != v and (u, v) not in sampled and (v, u) not in sampled:
                    sampled.add((u, v))
                attempts += 1

            for u, v in sampled:
                has_edge: int = 1 if (u, v) in edge_set else 0
                u_pos: int = dfg_nodes[u][1] + 1
                v_pos: int = dfg_nodes[v][1] + 1
                if u_pos >= max_seq_length or v_pos >= max_seq_length:
                    continue
                edge_pairs.append((i, u_pos, v_pos, has_edge))

        if edge_pairs:
            edge_batch_idx: torch.Tensor = torch.tensor(
                [p[0] for p in edge_pairs], dtype=torch.long
            )
            edge_node1_pos: torch.Tensor = torch.tensor(
                [p[1] for p in edge_pairs], dtype=torch.long
            )
            edge_node2_pos: torch.Tensor = torch.tensor(
                [p[2] for p in edge_pairs], dtype=torch.long
            )
            edge_labels: torch.Tensor = torch.tensor(
                [p[3] for p in edge_pairs], dtype=torch.float
            )
        else:
            edge_batch_idx = torch.tensor([], dtype=torch.long)
            edge_node1_pos = torch.tensor([], dtype=torch.long)
            edge_node2_pos = torch.tensor([], dtype=torch.long)
            edge_labels = torch.tensor([], dtype=torch.float)

        return {
            "input_ids": masked_ids,
            "attention_mask": attn_mask,
            "position_ids": pos_idx,
            "labels": labels,
            "edge_batch_idx": edge_batch_idx,
            "edge_node1_pos": edge_node1_pos,
            "edge_node2_pos": edge_node2_pos,
            "edge_labels": edge_labels,
        }
