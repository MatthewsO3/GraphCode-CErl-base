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
- [Pipeline (run.py)](#pipeline-runpy)
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
├── run.py              # End-to-end pipeline: setup → preprocess → train → evaluate
├── model.py            # Dataset class, GraphCodeBERTWithEdgePrediction, and data collator
├── train.py            # Training loop, optimizer setup, checkpointing, and metrics logging
├── evaluate.py         # UnifiedMLMEvaluator — DFG-aware evaluation across all 5 languages
├── preprocessing.py    # Tree-sitter DFG extraction and JSONL dataset builder
├── config.json         # All hyperparameters for training and evaluation
└── setup.py            # Dependency installer and environment checker
```

---

## Pipeline (run.py)

`run.py` is the recommended entry point. It runs all four stages in order — **setup → preprocess → train → evaluate** — reading every value from `config.json`, with any key overridable via CLI flag.

### Running the full pipeline

```bash
python run.py
```

### Overriding config values from the CLI

Any value from `config.json` can be overridden directly. The override is written back into `config.json` before sub-scripts run, so all scripts see the updated value.

```bash
# Change batch size and number of epochs
python run.py --batch_size 16 --epochs 3

# Preprocess only C++ with a smaller sample count
python run.py --lang cpp --max_samples 10000

# Point at a different Erlang file and output directory
python run.py --erlang_file data/my_erlang.jsonl --output_dir output/run_2
```

### Skipping or isolating stages

```bash
# Skip setup and preprocessing (e.g. data already prepared)
python run.py --skip setup preprocess

# Run only the evaluate stage
python run.py --only evaluate

# Run only training and evaluation
python run.py --only train evaluate
```

### Dry run

Print the exact subprocess commands that would be executed without running anything:

```bash
python run.py --dry_run
```

### Using a different config file

```bash
python run.py --config experiments/erlang_only.json
```

### CLI reference

| Flag | Affects | Description |
|---|---|---|
| `--skip STAGE [STAGE ...]` | pipeline | Stages to skip: `setup`, `preprocess`, `train`, `evaluate` |
| `--only STAGE [STAGE ...]` | pipeline | Run only these stages, skip all others |
| `--dry_run` | pipeline | Print commands without executing |
| `--config PATH` | pipeline | Path to config file (default: `config.json`) |
| `--lang` | preprocess | Language to preprocess: `cpp`, `python`, `java`, `javascript`, `all` |
| `--max_samples N` | preprocess | Max samples to collect per language |
| `--data_file PATH` | train | Path to training JSONL |
| `--output_dir PATH` | train | Output directory for checkpoints and metrics |
| `--checkpoint_path PATH` | train | Resume from an existing checkpoint |
| `--batch_size N` | train | Training batch size |
| `--epochs N` | train | Maximum training epochs |
| `--learning_rate F` | train | AdamW learning rate |
| `--max_length N` | train | Maximum sequence length |
| `--warmup_steps N` | train | LR scheduler warmup steps |
| `--mlm_probability F` | train | Token masking probability |
| `--validation_split F` | train | Fraction of data held out for validation |
| `--weight_decay F` | train | AdamW weight decay |
| `--early_stopping_patience N` | train | Epochs without improvement before stopping |
| `--model PATH` | evaluate | Model ID or path (defaults to `output_dir/best_model`) |
| `--mask_ratio F` | evaluate | Fraction of tokens to mask during evaluation |
| `--top_k N` | evaluate | Top-k predictions to retrieve |
| `--max_examples N` | evaluate | Max samples to evaluate per language |
| `--langs LANG [...]` | evaluate | Languages to evaluate |
| `--data_files PATH [...]` | evaluate | JSONL paths, parallel to `--langs` |

---

## Data Preprocessing

`preprocessing.py` downloads code from `codeparrot/github-code-clean` (streaming), extracts DFG edges via tree-sitter, tokenizes with the GraphCodeBERT tokenizer, and writes `.jsonl` files ready for training.

```bash
# Preprocess a single language
python preprocessing.py --lang cpp --max_samples 250000

# Preprocess all supported languages in one pass
python preprocessing.py --lang all --max_samples 250000
```

Supported values for `--lang`: `cpp`, `python`, `java`, `javascript`, `all`.

Output is written to `data/<lang>_processed.jsonl`. Each line is a JSON object with the following fields:

| Field | Description |
|---|---|
| `idx` | Unique sample identifier (`<lang>::<n>`) |
| `code` | Raw source code string |
| `code_tokens` | RoBERTa subword tokens |
| `dataflow_graph` | List of DFG edges: `(var_name, use_pos, "comesFrom", [var_name], [def_pos])` |
| `language` | Language string |

Samples are filtered to be between 100–10,000 characters and 3–500 lines, and must contain language-specific keywords (e.g. `std::`, `def `, `public `) to reduce noise.

### Erlang data

Erlang is not available on HuggingFace. Provide your own JSONL file and declare its path in `config.json` under `preprocess.erlang_file`. The file can contain either:

- **Pre-tokenized records** with a `"code_tokens"` field — used as-is.
- **Raw source records** with a `"code"` or `"source_code"` field — tokenized automatically using the GraphCodeBERT tokenizer. DFG extraction is skipped for Erlang (a warning is printed) since Tree-sitter Erlang may not be installed.

### Training data merge

After per-language preprocessing completes, `preprocessing.py` automatically merges the C++ and Erlang corpora into a single balanced, shuffled file at `data/train.jsonl`. Both corpora are truncated to the size of the smaller one so neither language dominates training.

This step is skipped if `preprocess.erlang_file` is not set in `config.json`.

---

## Training

Edit `config.json` to set `data_file` and `output_dir` under the `train` section, then run:

```bash
python train.py
```

All config values can be overridden via CLI flags:

```bash
python train.py \
    --data_file data/train.jsonl \
    --output_dir output/cpp_erl_run \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --max_length 256
```

To resume from a checkpoint:

```bash
python train.py \
    --checkpoint_path output/erlang_run/best_model \
    --data_file data/train.jsonl \
    --output_dir output/continued
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

```bash
python evaluate.py --config config.json
```

The evaluator reads `train.output_dir` from `config.json` and automatically uses `<output_dir>/best_model` as the model path. This can be overridden by setting `evaluate.model` explicitly.

### Output files

Results are saved inside `<output_dir>/best_model/`:

| File | Description |
|---|---|
| `eval_<lang>.jsonl` | Per-sample results + a final summary record for each language |
| `eval_summary.jsonl` | One-line-per-language aggregated summary across all evaluated languages |

Each `eval_<lang>.jsonl` contains one JSON record per evaluated sample (`"type": "sample"`) followed by a single summary record (`"type": "summary"`):

```jsonl
{"type": "sample", "language": "cpp", "top1_hits": 1, "top5_hits": 1, "total": 3, "mean_log_prob": -0.412}
{"type": "sample", "language": "cpp", "top1_hits": 2, "top5_hits": 3, "total": 4, "mean_log_prob": -0.287}
{"type": "summary", "language": "cpp", "top1_accuracy": 0.885, "top5_accuracy": 0.942, "perplexity": 1.95, "total_masked_positions": 18423, "top1_hits": 16304, "top5_hits": 17354}
```

Expected JSONL format for evaluation input (either works):

```jsonl
{"code_tokens": ["int", "Ġx", "Ġ=", "Ġ0", ";"], "dataflow_graph": [...]}
{"code": "int x = 0;"}
```

Metrics reported per language:

- **Top-1 Accuracy** — model's single best prediction matches the original token
- **Top-5 Accuracy** — correct token appears in the top 5 predictions
- **Perplexity** — exponentiated mean negative log-likelihood over masked positions

---

## Configuration Reference

`config.json` has three top-level sections:

### `preprocess`

| Key | Default | Description |
|---|---|---|
| `lang` | `"cpp"` | Language to preprocess: `cpp`, `python`, `java`, `javascript`, or `all` |
| `max_samples` | `10` | Maximum samples to collect per language |
| `erlang_file` | — | Path to a local Erlang JSONL file. Required to produce `data/train.jsonl` |

### `train`

| Key | Default | Description |
|---|---|---|
| `data_file` | — | Path to `.jsonl` training data |
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
| `model` | *(derived from `train.output_dir`)* | HuggingFace model ID or local path. Defaults to `<output_dir>/best_model` |
| `mask_ratio` | `0.15` | Fraction of tokens to mask during evaluation |
| `top_k` | `10` | Top-k predictions to retrieve |
| `max_examples` | `1000` | Maximum samples to evaluate per language |
| `langs` | `[python, java, javascript, cpp, erlang]` | Languages to evaluate |
| `data_files` | — | Parallel list of `.jsonl` paths, one per language |

### Example `config.json`

```json
{
  "preprocess": {
    "lang": "all",
    "max_samples": 250000,
    "erlang_file": "data/erlang_data.jsonl"
  },
  "train": {
    "data_file": "data/train.jsonl",
    "output_dir": "output/graphcode_cerl",
    "batch_size": 32,
    "epochs": 6,
    "learning_rate": 2e-5,
    "max_length": 256,
    "warmup_steps": 2000,
    "mlm_probability": 0.15,
    "validation_split": 0.15,
    "weight_decay": 0.01,
    "early_stopping_patience": 3
  },
  "evaluate": {
    "mask_ratio": 0.15,
    "top_k": 10,
    "max_examples": 2500,
    "langs": ["cpp", "erlang", "java", "python", "javascript"],
    "data_files": [
      "data/cpp_processed.jsonl",
      "data/erlang_data.jsonl",
      "data/java_processed.jsonl",
      "data/python_processed.jsonl",
      "data/javascript_processed.jsonl"
    ]
  }
}
```

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