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

# --- 1. Language Configuration Registry ---
# Configured for C++, Python, Java, and JavaScript
LANGUAGE_CONFIG = {
    'cpp': {
        'lib': 'tree_sitter_cpp',
        'def_nodes': ['declaration', 'init_declarator', 'parameter_declaration'],
        'assignment_nodes': ['assignment_expression'],
        'identifier_nodes': ['identifier', 'field_identifier'],
        'keywords': ['void ', 'int ', 'class ', 'std::'],
        'dataset_name': 'C++-all'
    },
    'python': {
        'lib': 'tree_sitter_python',
        'def_nodes': ['assignment', 'for_statement', 'function_definition', 'parameters'],
        'assignment_nodes': ['augmented_assignment'],
        'identifier_nodes': ['identifier'],
        'keywords': ['def ', 'import ', 'class ', 'self.'],
        'dataset_name': 'Python-all'
    },
    'java': {
        'lib': 'tree_sitter_java',
        'def_nodes': ['local_variable_declaration', 'formal_parameter', 'method_declaration'],
        'assignment_nodes': ['assignment_expression'],
        'identifier_nodes': ['identifier'],
        'keywords': ['public ', 'private ', 'class ', 'String '],
        'dataset_name': 'Java-all'
    },
    'javascript': {
        'lib': 'tree_sitter_javascript',
        'def_nodes': ['variable_declarator', 'formal_parameters', 'function_declaration'],
        'assignment_nodes': ['assignment_expression'],
        'identifier_nodes': ['identifier', 'variable'],
        'keywords': ['function ', 'const ', 'let ', 'var ', '=>'],
        'dataset_name': 'JavaScript-all'
    }
}

class UniversalPreprocessor:
    def __init__(self, lang_name: str):
        self.lang_name = lang_name.lower()
        if self.lang_name not in LANGUAGE_CONFIG:
            raise ValueError(f"Language {lang_name} not supported in registry.")
        
        self.config = LANGUAGE_CONFIG[self.lang_name]
        
        try:
            lang_module = __import__(self.config['lib'])
            
            # Create the Language object
            self.language = Language(lang_module.language())
            
            # Initialize the Parser directly with the language
            self.parser = Parser(self.language)
        except ImportError:
            raise ImportError(f"Please install {self.config['lib']} via pip.")

        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

    def is_definition(self, node) -> bool:
        parent = node.parent
        if not parent: return False
        
        # Standard definition logic for C-style and Scripting languages
        if parent.type in self.config['def_nodes']:
            return True
        
        if parent.type in self.config['assignment_nodes']:
            # Check if this node is the 'left' side (the variable being assigned)
            if hasattr(parent, 'child_by_field_name') and node == parent.child_by_field_name('left'):
                return True
        return False

    def extract_dfg(self, code_bytes: bytes, tree) -> List[Tuple]:
        root_node = tree.root_node
        var_definitions = defaultdict(list)
        var_uses = defaultdict(list)
        tokens = []
        node_to_token_pos = {}

        def collect_identifiers(node):
            if node.type in self.config['identifier_nodes']:
                if id(node) not in node_to_token_pos:
                    node_to_token_pos[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                collect_identifiers(child)

        collect_identifiers(root_node)

        def traverse(node):
            if node.type in self.config['identifier_nodes']:
                name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                pos = node_to_token_pos.get(id(node), -1)
                if pos != -1:
                    (var_definitions if self.is_definition(node) else var_uses)[name].append(pos)
            for child in node.children:
                traverse(child)

        traverse(root_node)

        dfg_edges = []
        for name, uses in var_uses.items():
            defs = sorted(var_definitions.get(name, []))
            for use_pos in uses:
                preceding = [d for d in defs if d < use_pos]
                if preceding:
                    # Link current use to the most recent definition
                    dfg_edges.append((name, use_pos, "comesFrom", [name], [preceding[-1]]))
        return dfg_edges

    def should_keep(self, code: str) -> bool:
        if not (100 < len(code) < 10000): return False
        if not (3 < code.count('\n') < 500): return False
        return any(kw in code for kw in self.config['keywords'])

    def process_sample(self, code: str, idx: int) -> Optional[Dict]:
        try:
            code_bytes = code.encode('utf8')
            tree = self.parser.parse(code_bytes)
            tokens = self.tokenizer.tokenize(code, add_prefix_space=True)

            if not (10 < len(tokens) < 450): return None

            dfg = self.extract_dfg(code_bytes, tree)
            if not dfg or len(dfg) < 2: return None

            return {
                'idx': f'{self.lang_name}::{idx}',
                'code': code,
                'code_tokens': tokens,
                'dataflow_graph': dfg,
                'language': self.lang_name
            }
        except: return None

def stream_dataset(lang: str, output_file: str, max_samples: int):
    preprocessor = UniversalPreprocessor(lang)
    dataset = load_dataset("codeparrot/github-code-clean", preprocessor.config['dataset_name'], split="train", streaming=True)

    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as f, tqdm(desc=f"Processing {lang}") as pbar:
        for example in dataset:
            if max_samples and processed_count >= max_samples: break
            code = example.get('code', "")
            if not preprocessor.should_keep(code): continue
            processed = preprocessor.process_sample(code, processed_count)
            if processed:
                f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                processed_count += 1
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='cpp', help='cpp, python, java, or javascript')
    parser.add_argument('--max_samples', type=int, default=10)
    args = parser.parse_args()
    output_path = f"data/{args.lang}_processed.jsonl"
    os.makedirs("data", exist_ok=True)
    stream_dataset(args.lang, output_path, args.max_samples)

if __name__ == "__main__":
    main()