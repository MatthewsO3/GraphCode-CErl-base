

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer


def find_project_root(start_path: Path = None) -> Path:
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current = start_path
    while True:
        config_path = current / 'config.json'
        if config_path.exists():
            return current

        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find project root. "
                "Make sure config.json exists in the project root directory."
            )
        current = parent


def load_config() -> Dict:
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


class GraphCodeBERTDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.max_length is None:
            try:
                config = load_config()
                self.max_length = config.get('train', {}).get('max_length', 512)
            except FileNotFoundError:
                self.max_length = 512

        self.samples = []
        print(f"Loading and processing data from {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.convert_sample_to_features(self.samples[idx])

    def convert_sample_to_features(self, sample: Dict) -> Dict:
        code_tokens = sample['code_tokens']
        dfg = sample.get('dataflow_graph', [])

        # ── Step 1: hard caps so nothing can ever exceed max_length ──────────
        # Layout: [CLS] + code(N) + [SEP] + dfg(D) + [SEP]  →  N + D + 3 <= max_length
        MAX_DFG   = min(64, self.max_length // 4)          # at most 25% of budget for DFG
        MAX_CODE  = self.max_length - MAX_DFG - 3          # remaining for code

        code_tokens = code_tokens[:MAX_CODE]
        valid_code_len = len(code_tokens)

        # ── Step 2: build DFG index from the (already truncated) token range ─
        adj = defaultdict(list)
        dfg_nodes, node_to_idx = [], {}

        for var, use_pos, _, _, dep_pos_list in dfg:
            if use_pos >= valid_code_len:          # out of range after code truncation
                continue
            if use_pos not in node_to_idx:
                node_to_idx[use_pos] = len(dfg_nodes)
                dfg_nodes.append((var, use_pos))
            use_idx = node_to_idx[use_pos]

            for def_pos in dep_pos_list:
                if def_pos >= valid_code_len:      # out of range
                    continue
                if def_pos not in node_to_idx:
                    node_to_idx[def_pos] = len(dfg_nodes)
                    dfg_nodes.append((var, def_pos))
                adj[use_idx].append(node_to_idx[def_pos])

        # ── Step 3: cap DFG node count ────────────────────────────────────────
        if len(dfg_nodes) > MAX_DFG:
            keep = set(range(MAX_DFG))
            dfg_nodes = dfg_nodes[:MAX_DFG]
            adj = defaultdict(list, {
                i: [j for j in adjs if j in keep]
                for i, adjs in adj.items()
                if i in keep
            })

        dfg_token_count = len(dfg_nodes)

        # ── Step 4: final sanity check on total length ────────────────────────
        total = valid_code_len + dfg_token_count + 3      # +3 for CLS, SEP, SEP
        assert total <= self.max_length, \
            f"Still too long after capping: code={valid_code_len} dfg={dfg_token_count} total={total}"

        # ── Step 5: build token sequence ─────────────────────────────────────
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        dfg_start_pos = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * dfg_token_count)
        tokens.append(self.tokenizer.sep_token)

        input_ids   = self.tokenizer.convert_tokens_to_ids(tokens)
        position_idx = (list(range(valid_code_len + 2))
                        + [0] * dfg_token_count
                        + [valid_code_len + 2])

        padding_len = self.max_length - len(input_ids)
        input_ids    += [self.tokenizer.pad_token_id] * padding_len
        position_idx += [0] * padding_len

        # ── Step 6: attention mask ────────────────────────────────────────────
        attn_mask = np.zeros((self.max_length, self.max_length), dtype=np.bool_)
        code_len  = valid_code_len + 2          # CLS + code + SEP

        attn_mask[:code_len, :code_len] = True  # code attends to code

        for i in range(len(tokens)):            # every real token attends to itself
            attn_mask[i, i] = True

        for i, (_, code_pos) in enumerate(dfg_nodes):
            dfg_abs  = dfg_start_pos + i
            code_abs = code_pos + 1             # +1 for CLS
            attn_mask[dfg_abs, code_abs] = True
            attn_mask[code_abs, dfg_abs] = True

        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start_pos + i, dfg_start_pos + j
                attn_mask[u, v] = True
                attn_mask[v, u] = True

        assert len(input_ids)  == self.max_length
        assert len(position_idx) == self.max_length
        assert attn_mask.shape == (self.max_length, self.max_length)

        return {
            'input_ids':      torch.tensor(input_ids,   dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask,   dtype=torch.bool),
            'position_idx':   torch.tensor(position_idx, dtype=torch.long),
            'dfg_info': {
                'nodes': dfg_nodes,
                'edges': [(i, j) for i, adjs in adj.items() for j in adjs]
            }
        }

class GraphCodeBERTWithEdgePrediction(nn.Module):
    """
    Unified GraphCodeBERT model accepting both Erlang and C++ input formats.

    ERLANG format inputs:
    - position_idx: [batch, seq_len]
    - edge_candidates: [batch, max_edges, 2]
    - edge_labels: [batch, max_edges]
    - alignment_candidates, alignment_labels (optional)
    - dfg_start_idx

    C++ format inputs:
    - position_ids: [batch, seq_len]
    - edge_batch_idx, edge_node1_pos, edge_node2_pos, edge_labels

    Returns:
    - dict with keys: loss, mlm_loss, edge_loss
    """

    def __init__(self, base_model_name: Optional[str] = None):
        super().__init__()

        if base_model_name is None:
            try:
                config = load_config()
                base_model_name = config.get('model', {}).get('base_model', 'microsoft/graphcodebert-base')
            except FileNotFoundError:
                base_model_name = 'microsoft/graphcodebert-base'

        self.roberta_mlm = RobertaForMaskedLM.from_pretrained(base_model_name)
        hidden_size = self.roberta_mlm.config.hidden_size

        try:
            config = load_config()
            hidden_dropout = config.get('model', {}).get('hidden_dropout_prob', 0.2)
            attention_dropout = config.get('model', {}).get('attention_probs_dropout_prob', 0.2)
        except FileNotFoundError:
            hidden_dropout = 0.2
            attention_dropout = 0.2

        self.roberta_mlm.config.hidden_dropout_prob = hidden_dropout
        self.roberta_mlm.config.attention_probs_dropout_prob = attention_dropout

        # Edge classifier for predicting DFG edges
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,

            # C++ format
            position_ids=None,
            edge_batch_idx=None,
            edge_node1_pos=None,
            edge_node2_pos=None,

            # Erlang format
            position_idx=None,
            edge_candidates=None,
            alignment_candidates=None,
            alignment_labels=None,
            dfg_start_idx=None,

            edge_labels=None,
            **kwargs
    ):
        """
        Flexible forward pass accepting both Erlang and C++ input formats.

        Automatically detects format and converts as needed.
        """

        # Step 1: Normalize position input
        if position_ids is None and position_idx is not None:
            position_ids = position_idx
        elif position_ids is None:
            # Generate default positions
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Step 2: Convert edge format if needed
        # If Erlang format is provided (edge_candidates), convert to C++ format
        if edge_candidates is not None and edge_batch_idx is None:
            edge_batch_idx, edge_node1_pos, edge_node2_pos = \
                self._convert_erlang_edges_to_cpp(edge_candidates)

        # Step 3: MLM forward pass
        mlm_outputs = self.roberta_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True
        )
        mlm_loss = mlm_outputs.loss if labels is not None else None

        # Step 4: Edge prediction
        edge_loss = None
        if (edge_batch_idx is not None and len(edge_batch_idx) > 0 and
                edge_node1_pos is not None and edge_node2_pos is not None and edge_labels is not None):
            hidden_states = mlm_outputs.hidden_states[-1]

            node1_repr = hidden_states[edge_batch_idx, edge_node1_pos]
            node2_repr = hidden_states[edge_batch_idx, edge_node2_pos]
            edge_repr = torch.cat([node1_repr, node2_repr], dim=-1)
            edge_logits = self.edge_classifier(edge_repr).squeeze(-1)

            # Ensure edge_labels is float for BCE loss
            edge_labels_float = edge_labels.float() if edge_labels.dtype != torch.float32 else edge_labels
            edge_loss = nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_labels_float)

        # Step 5: Combine losses
        if mlm_loss is not None and edge_loss is not None:
            total_loss = mlm_loss + edge_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif edge_loss is not None:
            total_loss = edge_loss
        else:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        # Return as dict for consistency
        return {
            'loss': total_loss,
            'mlm_loss': mlm_loss,
            'edge_loss': edge_loss,
        }

    def _convert_erlang_edges_to_cpp(self, edge_candidates: torch.Tensor) -> tuple:
        """
        Convert Erlang edge format to C++ edge format.

        Erlang: [batch_size, max_edges, 2] → each [..., :2] contains [node1_pos, node2_pos]
        C++: Returns (edge_batch_idx, edge_node1_pos, edge_node2_pos)
        """
        batch_size = edge_candidates.shape[0]

        batch_indices = []
        node1_positions = []
        node2_positions = []

        for batch_idx in range(batch_size):
            edges = edge_candidates[batch_idx]  # [max_edges, 2]

            # Filter out padding (assume padding is [0, 0])
            valid_edges = edges[(edges[:, 0] != 0) | (edges[:, 1] != 0)]

            if len(valid_edges) > 0:
                batch_indices.extend([batch_idx] * len(valid_edges))
                node1_positions.extend(valid_edges[:, 0].tolist())
                node2_positions.extend(valid_edges[:, 1].tolist())

        device = edge_candidates.device

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

    def save_pretrained(self, save_directory):
        """
        Save only the base model (roberta_mlm), not the edge_classifier.
        This makes it compatible with HuggingFace's from_pretrained() and
        allows Erlang script to load it.
        """
        self.roberta_mlm.save_pretrained(save_directory)
        print(f"✓ Base model saved to {save_directory}")
        print(f"  Note: edge_classifier is NOT saved (it gets re-initialized on load)")

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Load model from pretrained checkpoint.
        Creates instance with base model, edge_classifier gets fresh initialization.
        """
        instance = cls(model_name_or_path)
        return instance


@dataclass
class MLMWithEdgePredictionCollator:
    tokenizer: RobertaTokenizer
    mlm_probability: Optional[float] = None
    edge_sample_ratio: Optional[float] = None

    def __post_init__(self):
        if self.mlm_probability is None or self.edge_sample_ratio is None:
            try:
                config = load_config()
                train_config = config.get('train', {})
                model_config = config.get('model', {})

                if self.mlm_probability is None:
                    self.mlm_probability = train_config.get('mlm_probability', 0.15)
                if self.edge_sample_ratio is None:
                    self.edge_sample_ratio = model_config.get('edge_sample_ratio', 0.3)
            except FileNotFoundError:
                if self.mlm_probability is None:
                    self.mlm_probability = 0.15
                if self.edge_sample_ratio is None:
                    self.edge_sample_ratio = 0.3

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        max_seq_length = examples[0]['input_ids'].shape[0]

        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attn_mask = torch.stack([ex['attention_mask'] for ex in examples])
        pos_idx = torch.stack([ex['position_idx'] for ex in examples])

        assert input_ids.shape == (batch_size, max_seq_length)
        assert attn_mask.shape == (batch_size, max_seq_length, max_seq_length)
        assert pos_idx.shape == (batch_size, max_seq_length)

        labels, masked_ids = input_ids.clone(), input_ids.clone()
        for i in range(batch_size):
            code_indices = (pos_idx[i] > 1).nonzero(as_tuple=True)[0]
            if len(code_indices) > 1:
                code_indices = code_indices[:-1]
            if len(code_indices) == 0:
                continue
            num_mask = max(1, int(len(code_indices) * self.mlm_probability))
            mask_pos = code_indices[torch.randperm(len(code_indices))[:num_mask]]
            for pos in mask_pos:
                if random.random() < 0.8:
                    masked_ids[i, pos] = self.tokenizer.mask_token_id
                elif random.random() < 0.5:
                    masked_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)
            mask_ind = torch.zeros_like(labels[i], dtype=torch.bool)
            mask_ind[mask_pos] = True
            labels[i, ~mask_ind] = -100
        labels[masked_ids == self.tokenizer.pad_token_id] = -100

        edge_pairs = []
        max_pairs = 20
        for i in range(batch_size):
            if 'dfg_info' not in examples[i]:
                continue
            dfg_nodes = examples[i]['dfg_info']['nodes']
            dfg_edges = examples[i]['dfg_info']['edges']
            if len(dfg_nodes) < 2:
                continue

            edge_set = set(dfg_edges)
            edge_set.update((v, u) for u, v in dfg_edges)

            num_nodes = len(dfg_nodes)
            num_pairs = min(max_pairs, int(num_nodes * (num_nodes - 1) / 2 * self.edge_sample_ratio))
            sampled = set()
            attempts = 0
            while len(sampled) < num_pairs and attempts < num_pairs * 3:
                u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                if u != v and (u, v) not in sampled and (v, u) not in sampled:
                    sampled.add((u, v))
                attempts += 1

            for u, v in sampled:
                has_edge = 1 if (u, v) in edge_set else 0
                u_pos = dfg_nodes[u][1] + 1
                v_pos = dfg_nodes[v][1] + 1
                # ✅ FIX: Skip if positions exceed sequence length
                if u_pos >= max_seq_length or v_pos >= max_seq_length:
                    continue
                edge_pairs.append((i, u_pos, v_pos, has_edge))

        if edge_pairs:
            edge_batch_idx = torch.tensor([p[0] for p in edge_pairs], dtype=torch.long)
            edge_node1_pos = torch.tensor([p[1] for p in edge_pairs], dtype=torch.long)
            edge_node2_pos = torch.tensor([p[2] for p in edge_pairs], dtype=torch.long)
            edge_labels = torch.tensor([p[3] for p in edge_pairs], dtype=torch.float)
        else:
            edge_batch_idx = torch.tensor([], dtype=torch.long)
            edge_node1_pos = torch.tensor([], dtype=torch.long)
            edge_node2_pos = torch.tensor([], dtype=torch.long)
            edge_labels = torch.tensor([], dtype=torch.float)

        return {
            'input_ids': masked_ids,
            'attention_mask': attn_mask,
            'position_ids': pos_idx,  # C++ script expects position_ids
            'labels': labels,
            'edge_batch_idx': edge_batch_idx,
            'edge_node1_pos': edge_node1_pos,
            'edge_node2_pos': edge_node2_pos,
            'edge_labels': edge_labels
        }