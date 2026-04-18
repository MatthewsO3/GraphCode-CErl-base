# GraphCode-CErl-base

A fine-tuned version of [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base) for Masked Language Modeling (MLM) on **C++ and Erlang** source code, with zero-shot evaluation on Python, Java, and JavaScript.

The model extends GraphCodeBERT's pre-training with a combined MLM + DFG edge-prediction objective, enabling it to reason about both the syntactic structure and the semantic data-flow relationships between variables in code.

**HuggingFace model:** [`MatthewsO3/GraphCode-CErl-base`](https://huggingface.co/MatthewsO3/GraphCode-CErl-base)

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full DFG-Based Inference](#full-dfg-based-inference)
- [Repository Structure](#repository-structure)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

---

## Overview

GraphCodeBERT takes as input a concatenation of (1) code token sequences and (2) data flow graph (DFG) node representations, linked by a 2D attention mask. This allows the model to attend across both the token stream and the variable-level semantic graph simultaneously.

This fine-tuning extends the base model with:

- **Continued MLM pre-training** on C++ and Erlang corpora (250k samples each)
- **DFG edge-prediction loss** — a binary cross-entropy objective that predicts whether two DFG nodes share a data-flow edge, applied jointly during training
- **Tree-sitter DFG extraction** for C++, Erlang, Python, Java, and JavaScript, enabling evaluation on all five languages without requiring pre-tokenized data

The model converges to a validation loss of **0.3701** at epoch 5 and achieves strong token-level accuracy on both training languages, with meaningful zero-shot transfer to Java.

---

## Results

| Language | Top-1 Acc | Top-5 Acc | Perplexity | Notes |
|---|---|---|---|---|
| C++ | 88.5% | 94.2% | ~1.95 | Trained |
| Erlang | 86.5% | 93.1% | ~2.05 | Trained |
| Java | 83.5% | 91.5% | ~2.55 | Zero-shot |
| Python | 77.8% | 88.6% | ~3.30 | Zero-shot |
| JavaScript | 76.2% | 88.6% | ~3.35 | Zero-shot |

Evaluation uses a mask ratio of 0.15 on up to 2,500 samples per language. Full training loss curve in the [model card](https://huggingface.co/MatthewsO3/GraphCode-CErl-base).

---

## Installation

**Requirements:** Python 3.9+, pip

Run the provided setup script to install all dependencies and verify your environment:

```bash
python setup.py
```

This installs PyTorch, Transformers, tree-sitter parsers, and the HuggingFace `datasets` library, then runs a self-check on your GPU and tokenizer.

Alternatively, install manually:

```bash
pip install torch==2.10.0 transformers==4.57.6 numpy==2.4.1 tqdm==4.67.1 \
    tree-sitter==0.25.2 tree-sitter-java==0.23.5 tree-sitter-python==0.25.0 \
    tree-sitter-javascript==0.25.0 tree-sitter-cpp==0.23.4 datasets==3.3.2
```

> **Note on Erlang:** `tree-sitter-erlang` is not available on PyPI. Install it manually from source:
> ```bash
> pip install https://github.com/the-mikedavis/tree-sitter-erlang/tarball/master
> ```
> Without it, Erlang evaluation falls back to tokenizer-only mode (no DFG edges).

---

## Quick Start

The simplest way to use the model is through the HuggingFace `fill-mask` pipeline. This does **not** use DFG inputs — it treats the model as a standard masked language model.

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="MatthewsO3/GraphCode-CErl-base")

# C++ example
results = fill("int <mask> = 0;")
for r in results[:3]:
    print(r['token_str'], f"({r['score']:.2%})")

# Erlang example
results = fill("F = fun(<mask>) -> ok end.")
for r in results[:3]:
    print(r['token_str'], f"({r['score']:.2%})")
```

For more detailed token-level prediction with logits:

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaForMaskedLM.from_pretrained("MatthewsO3/GraphCode-CErl-base")
model.eval()

code = "std::vector<int> <mask>(10, 0);"
inputs = tokenizer(code, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
top5 = logits[0, mask_idx].topk(5).indices
print([tokenizer.decode(i) for i in top5[0]])
```

---

## Full DFG-Based Inference

For graph-aware inference (the same mode used during training and evaluation), use the `UnifiedMLMEvaluator` from `evaluate.py` directly. This extracts the data flow graph via tree-sitter and builds the full 2D attention mask before running the model.

```python
from evaluate import UnifiedMLMEvaluator

evaluator = UnifiedMLMEvaluator(
    model_path="MatthewsO3/GraphCode-CErl-base",
    max_seq_length=256
)

sample = {
    "code": """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""
}

result = evaluator.evaluate_sample(
    sample=sample,
    lang="cpp",
    mask_ratio=0.15,
    top_k=10
)

if result:
    print(f"Top-1 correct: {result['t1']} / {result['total']}")
    print(f"Top-5 correct: {result['t5']} / {result['total']}")
```

Supported `lang` values: `"python"`, `"java"`, `"javascript"`, `"cpp"`, `"erlang"`.

You can also pass pre-tokenized JSONL samples (produced by `preprocessing.py`) directly:

```python
import json

with open("data/cpp.jsonl") as f:
    sample = json.loads(f.readline())

# sample already contains 'code_tokens' and 'dataflow_graph'
result = evaluator.evaluate_sample(sample, lang="cpp", mask_ratio=0.15, top_k=10)
```

---

## Repository Structure

```
GraphCode-CErl-base/
├── model.py            # Dataset class, GraphCodeBERTWithEdgePrediction, and data collator
├── train.py            # Training loop, optimizer setup, checkpointing, and metrics logging
├── evaluate.py         # UnifiedMLMEvaluator — DFG-aware evaluation across all 5 languages
├── preprocessing.py    # Tree-sitter DFG extraction and JSONL dataset builder
├── config.json         # All hyperparameters for training and evaluation
└── setup.py            # Dependency installer and environment checker
```

---

## Data Preprocessing

`preprocessing.py` downloads code from `codeparrot/github-code-clean` (streaming), extracts DFG edges via tree-sitter, tokenizes with the GraphCodeBERT tokenizer, and writes `.jsonl` files ready for training.

```bash
# Preprocess C++ (default)
python preprocessing.py --lang cpp --max_samples 250000

# Preprocess Python
python preprocessing.py --lang python --max_samples 250000

# Supported: cpp, python, java, javascript
```

Output is written to `data/<lang>_processed.jsonl`. Each line is a JSON object with the following fields:

| Field | Description |
|---|---|
| `idx` | Unique sample identifier (`<lang>::<n>`) |
| `code` | Raw source code string |
| `code_tokens` | RoBERTa subword tokens |
| `dataflow_graph` | List of DFG edges: `(var_name, use_pos, "comesFrom", [var_name], [def_pos])` |
| `language` | Language string |

> **Note:** Erlang training data was collected via a custom scraper and is not publicly available. For evaluation, provide your own `.jsonl` file following the schema above, or pass raw source code dicts with a `"code"` key.

Samples are filtered to be between 100–10,000 characters and 3–500 lines, and must contain language-specific keywords (e.g. `std::`, `def `, `public `) to reduce noise.

---

## Training

Edit `config.json` to point `data_file` at your preprocessed `.jsonl` and `output_dir` at your desired output location, then run:

```bash
python train.py
```

All config values can be overridden via CLI flags:

```bash
python train.py \
    --data_file data/cpp_processed.jsonl \
    --output_dir output/cpp_run \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --max_length 256
```

To resume from a checkpoint (e.g. a model previously trained on Erlang):

```bash
python train.py \
    --checkpoint_path output/erlang_run/best_model \
    --data_file data/cpp_processed.jsonl \
    --output_dir output/cpp_continued
```

### What gets saved

After each epoch, the trainer saves:

- `output_dir/checkpoints/epoch_NNN/` — full model checkpoint (HuggingFace format)
- `output_dir/best_model/` — best checkpoint by validation loss
- `output_dir/training_history.json` — per-batch and per-epoch loss curves
- `output_dir/training_summary.json` — high-level summary statistics
- `output_dir/training_metrics.csv` — epoch-level metrics table

Training uses mixed-precision (AMP) automatically on CUDA GPUs and fp32 on CPU/MPS. Early stopping triggers after 3 epochs with no improvement in validation loss (configurable via `early_stopping_patience`).

### Training objective

The loss is a sum of two terms:

1. **MLM loss** — cross-entropy over randomly masked code tokens (15% masking probability, applied only to code positions, not DFG nodes)
2. **Edge prediction loss** — binary cross-entropy over sampled DFG node pairs, predicting whether a data-flow edge exists between them

---

## Evaluation

Run evaluation on any combination of languages and data files:

```bash
python evaluate.py --config config.json
```

The `evaluate` section of `config.json` controls which model, languages, and data files are used. You can also point it at raw source code files — the evaluator will tokenize and extract DFG on the fly if `code_tokens` is absent from the JSONL.

Expected JSONL format for evaluation (either works):

```jsonl
# Pre-tokenized (faster)
{"code_tokens": ["int", "Ġx", "Ġ=", "Ġ0", ";"], "dataflow_graph": [...]}

# Raw source (DFG extracted at eval time, requires tree-sitter)
{"code": "int x = 0;"}
```

Metrics reported per language:

- **Top-1 Accuracy** — model's single best prediction matches the original token
- **Top-5 Accuracy** — correct token appears in the top 5 predictions
- **Perplexity** — exponentiated mean negative log-likelihood over masked positions

---

## Configuration Reference

`config.json` has two top-level sections:

### `train`

| Key | Default | Description |
|---|---|---|
| `data_file` | — | Path to `.jsonl` training data (relative to project root) |
| `output_dir` | — | Output directory for checkpoints and metrics |
| `checkpoint_path` | `""` | Path to a checkpoint to resume from (empty = start from base model) |
| `batch_size` | `32` | Training batch size |
| `epochs` | `6` | Maximum number of training epochs |
| `learning_rate` | `2e-5` | AdamW learning rate |
| `max_length` | `256` | Maximum sequence length (code + DFG tokens combined) |
| `warmup_steps` | `2000` | Linear warmup steps for the LR scheduler |
| `mlm_probability` | `0.15` | Fraction of code tokens to mask per sample |
| `validation_split` | `0.15` | Fraction of data held out for validation |
| `weight_decay` | `0.01` | AdamW weight decay |
| `early_stopping_patience` | `3` | Epochs without improvement before stopping |

### `evaluate`

| Key | Default | Description |
|---|---|---|
| `model` | `MatthewsO3/GraphCode-CErl-base` | HuggingFace model ID or local path |
| `mask_ratio` | `0.15` | Fraction of tokens to mask during evaluation |
| `top_k` | `10` | Top-k predictions to retrieve |
| `max_examples` | `1000` | Maximum samples to evaluate per language |
| `langs` | `[python, java, javascript, cpp, erlang]` | Languages to evaluate |
| `data_files` | — | Parallel list of `.jsonl` paths, one per language |

---

## Citation

If you use this model, please also cite the original GraphCodeBERT paper:

```bibtex
@inproceedings{guo2021graphcodebert,
  title     = {GraphCodeBERT: Pre-training Code Representations with Data Flow},
  author    = {Guo, Daya and Ren, Shuo and Lu, Shuai and Feng, Zhangyin and Tang, Duyu
               and Liu, Shujie and Zhou, Long and Duan, Nan and Svyatkovskiy, Alexey
               and Fu, Shengyu and Tufano, Michele and Deng, Shao Kun and Clement, Colin
               and Drain, Dawn and Sundaresan, Neel and Yin, Jian and Jiang, Daxin
               and Zhou, Ming},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}
```

---

## License

MIT
