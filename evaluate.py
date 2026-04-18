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

# --- Tree-sitter Dynamic Imports ---
SUPPORTED_TS_LANGS = {}
try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjs
    import tree_sitter_python as tspy
    import tree_sitter_cpp as tscpp
    # Note: tree_sitter_erlang requires the library to be installed
    import tree_sitter_erlang as tserl 
    
    SUPPORTED_TS_LANGS = {
        'java': Language(tsjava.language()),
        'javascript': Language(tsjs.language()),
        'python': Language(tspy.language()),
        'cpp': Language(tscpp.language()),
        'erlang': Language(tserl.language())
    }
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    print("Warning: Some Tree-sitter parsers not found. Source-code evaluation limited to installed parsers.")

random.seed(42)
torch.manual_seed(42)

class UnifiedMLMEvaluator:
    def __init__(self, model_path: str, device: str = None, max_seq_length: int = 512):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_seq_length = max_seq_length
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        
        print(f"Loading model from {model_path}...")
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        self.parser = Parser() if TS_AVAILABLE else None

    def get_dfg_from_source(self, code: str, lang: str) -> List[Tuple]:
        """Extract DFG using Tree-sitter for Java, JS, Python, C++, or Erlang."""
        if not TS_AVAILABLE or lang not in SUPPORTED_TS_LANGS:
            return []
        
        self.parser.set_language(SUPPORTED_TS_LANGS[lang])
        code_bytes = code.encode('utf8')
        tree = self.parser.parse(code_bytes)
        root = tree.root_node
        
        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type in ['identifier', 'variable']:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children: find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            if not p: return False
            # Logic for existing and new languages
            if lang == 'java':
                return p.type in ['local_variable_declaration', 'formal_parameter', 'method_declaration'] or \
                       (p.type == 'assignment_expression' and node == p.child_by_field_name('left'))
            if lang == 'javascript':
                return p.type in ['variable_declarator', 'formal_parameters', 'function_declaration'] or \
                       (p.type == 'assignment_expression' and node == p.child_by_field_name('left'))
            if lang == 'python':
                return p.type in ['assignment', 'for_statement', 'function_definition', 'parameters'] or \
                       (p.type == 'augmented_assignment' and node == p.child_by_field_name('left'))
            if lang == 'cpp':
                return p.type in ['declaration', 'parameter_declaration'] or \
                       (p.type == 'assignment_expression' and node == p.child_by_field_name('left'))
            if lang == 'erlang':
                return p.type in ['variable'] and p.parent.type in ['match_expression', 'clause']
            return False

        def find_vars(node):
            if node.type in ['identifier', 'variable']:
                name = code_bytes[node.start_byte:node.end_byte].decode('utf8', 'ignore')
                pos = node_map.get(id(node), -1)
                if pos != -1:
                    (defs if is_def(node) else uses)[name].append(pos)
            for child in node.children: find_vars(child)

        find_vars(root)
        edges = []
        for name, use_positions in uses.items():
            def_positions = sorted(defs.get(name, []))
            for use_pos in use_positions:
                preds = [d for d in def_positions if d < use_pos]
                if preds: edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))
        return edges

    def build_inputs(self, masked_tokens: List[str], dfg: List, max_length: int) -> Dict:
        """Logic preserved from model.py and mixed_eval.py."""
        MAX_DFG = min(64, max_length // 4)
        MAX_CODE = max_length - MAX_DFG - 3
        
        masked_tokens = masked_tokens[:MAX_CODE]
        valid_code_len = len(masked_tokens)

        adj, nodes, node_map = defaultdict(list), [], {}
        for edge in dfg:
            var, use_pos = edge[0], edge[1]
            dep_pos_list = edge[4]
            if use_pos >= valid_code_len: continue
            if use_pos not in node_map:
                node_map[use_pos] = len(nodes)
                nodes.append((var, use_pos))
            use_idx = node_map[use_pos]
            for def_pos in dep_pos_list:
                if def_pos >= valid_code_len: continue
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
        pos_ids = list(range(valid_code_len + 2)) + [0] * len(nodes) + [valid_code_len + 2]
        
        mask = np.zeros((len(ids), len(ids)), dtype=bool)
        code_len = valid_code_len + 2
        mask[:code_len, :code_len] = True
        for i in range(len(ids)): mask[i, i] = True
        for i, (_, code_pos) in enumerate(nodes):
            dfg_abs, code_abs = dfg_start + i, code_pos + 1
            mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start + i, dfg_start + j
                mask[u, v] = mask[v, u] = True

        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([mask.tolist()]),
            'position_ids': torch.tensor([pos_ids])
        }

    def evaluate_sample(self, sample: Dict, lang: str, mask_ratio: float, top_k: int) -> Optional[Dict]:
        """Unifies pre-tokenized JSONL logic and raw source code logic."""
        if 'code_tokens' in sample:
            code_tokens = sample['code_tokens']
            dfg = sample.get('dataflow_graph', [])
        else:
            code = sample.get('code') or sample.get('source_code', "")
            code_tokens = self.tokenizer.tokenize(code)
            dfg = self.get_dfg_from_source(code, lang)
        MAX_DFG = min(64, self.max_seq_length // 4)
        MAX_CODE = self.max_seq_length - MAX_DFG - 3
        
        # Truncate code_tokens immediately so candidate_positions are always valid
        code_tokens = code_tokens[:MAX_CODE]
        code_tokens = [t for t in code_tokens if t not in (self.tokenizer.cls_token, self.tokenizer.sep_token)]
        
        # Filtering logic for comparable perplexity
        candidate_positions = [i for i, t in enumerate(code_tokens) 
                               if len(t.replace("Ġ", "").replace("Ċ", "").replace("Â", "")) > 1]
        
        if not candidate_positions: return None
        num_mask = max(1, int(len(candidate_positions) * mask_ratio))
        mask_pos = sorted(random.sample(candidate_positions, min(num_mask, len(candidate_positions))))

        orig_ids = []
        for i in mask_pos:
            tid = self.tokenizer.convert_tokens_to_ids(code_tokens[i])
            orig_ids.append(tid if tid != self.tokenizer.unk_token_id else None)

        masked_tokens = code_tokens.copy()
        for pos in mask_pos: masked_tokens[pos] = self.tokenizer.mask_token

        inputs = self.build_inputs(masked_tokens, dfg, self.max_seq_length)
        
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

        results = {'t1': 0, 't5': 0, 'lp': [], 'total': 0}
        for i, pos in enumerate(mask_pos):
            if orig_ids[i] is None: continue
            
            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            target_id = orig_ids[i]
            
            _, top_indices = torch.topk(probs, top_k)
            if target_id == top_indices[0]: results['t1'] += 1
            if target_id in top_indices[:5]: results['t5'] += 1
            
            results['lp'].append(np.log(max(probs[target_id].item(), 1e-9)))
            results['total'] += 1

        return results if results['total'] > 0 else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
        eval_config = full_config.get('evaluate', {})
        max_length = full_config.get('train', {}).get('max_length', 512)

    model_path = eval_config.get('model')
    langs = eval_config.get('langs', [])
    data_files = eval_config.get('data_files', [])
    mask_ratio = eval_config.get('mask_ratio', 0.15)
    top_k = eval_config.get('top_k', 10)
    max_ex = eval_config.get('max_examples', 1000)

    evaluator = UnifiedMLMEvaluator(model_path, max_seq_length=max_length)
    
    all_metrics = []
    for file_path, lang in zip(data_files, langs):
        print(f"\n--- Evaluating {lang} from {file_path} ---")
        metrics = {'t1': 0, 't5': 0, 'total': 0, 'lp': []}
        
        if not os.path.exists(file_path): continue

        with open(file_path, 'r') as f:
            lines = f.readlines()
            if max_ex: lines = lines[:max_ex]
            
            for line in tqdm(lines, desc=f"Processing {lang}"):
                sample = json.loads(line)
                res = evaluator.evaluate_sample(sample, lang.lower(), mask_ratio, top_k)
                if res:
                    metrics['t1'] += res['t1']
                    metrics['t5'] += res['t5']
                    metrics['total'] += res['total']
                    metrics['lp'].extend(res['lp'])

        if metrics['total'] > 0:
            ppl = np.exp(-np.mean(metrics['lp']))
            print(f"Results for {lang}: Top-1: {metrics['t1']/metrics['total']:.2%}, PPL: {ppl:.4f}")
            all_metrics.append(metrics)

if __name__ == "__main__":
    main()