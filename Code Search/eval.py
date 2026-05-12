import argparse
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

#     python eval.py --model_dir "cs_models/best_model" --eval_file "data/code_search_eval.jsonl" --distractors_file "data/combined_distractors.jsonl"
from model import CodeSearchModel

def load_jsonl(filepath):
    """Utility to load JSONL files."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_embeddings(texts, model_fn, tokenizer, max_length, batch_size, device):
    """Generate embeddings for a list of texts in batches."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        
        # Ensure batch_texts are strings
        batch_texts = [str(text) if text is not None else "" for text in batch_texts]
        
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Pass inputs to the respective model function (nl_inputs or code_inputs)
            embeddings = model_fn(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            )
            
            all_embeddings.append(embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)

    model = CodeSearchModel(base_model).to(device)
    model.eval()

    print("Loading evaluation and distractor datasets...")
    eval_data = load_jsonl(args.eval_file)
    distractor_data = load_jsonl(args.distractors_file)

    queries = [item['positive'] for item in eval_data]
    correct_code = [item['code'] for item in eval_data]
    distractor_code = [item['code'] for item in distractor_data]
    code_corpus = correct_code + distractor_code

    print(f"Total Queries: {len(queries)}")
    print(f"Total Code Corpus: {len(code_corpus)} ({len(correct_code)} targets + {len(distractor_code)} distractors)")

    print("\nGenerating Natural Language Query Embeddings...")
    query_embeddings = get_embeddings(
        texts=queries,
        model_fn=lambda ids, attention_mask: model(nl_inputs=ids, attention_mask=attention_mask),
        tokenizer=tokenizer,
        max_length=args.nl_len,
        batch_size=args.batch_size,
        device=device
    )

    print("\nGenerating Code Corpus Embeddings...")
    code_embeddings = get_embeddings(
        texts=code_corpus,
        model_fn=lambda ids, attention_mask: model(code_inputs=ids, attention_mask=attention_mask),
        tokenizer=tokenizer,
        max_length=args.code_len,
        batch_size=args.batch_size,
        device=device
    )

    print("\nCalculating metrics...")

    query_embeddings = query_embeddings.to(device)
    code_embeddings = code_embeddings.to(device)

    similarities = torch.matmul(query_embeddings, code_embeddings.T)

    mrr = 0.0
    r1, r5, r10 = 0, 0, 0
    num_queries = len(queries)

    lang_stats = {}

    for i in range(num_queries):
        scores = similarities[i]
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        correct_index = i
        rank = ranked_indices.index(correct_index) + 1

        mrr += 1.0 / rank
        if rank <= 1:  r1 += 1
        if rank <= 5:  r5 += 1
        if rank <= 10: r10 += 1

        lang = eval_data[i].get("language", "unknown")
        if lang not in lang_stats:
            lang_stats[lang] = {"mrr": 0.0, "r1": 0, "r5": 0, "r10": 0, "count": 0}
        lang_stats[lang]["mrr"] += 1.0 / rank
        lang_stats[lang]["count"] += 1
        if rank <= 1:  lang_stats[lang]["r1"] += 1
        if rank <= 5:  lang_stats[lang]["r5"] += 1
        if rank <= 10: lang_stats[lang]["r10"] += 1

    mrr /= num_queries
    r1 /= num_queries
    r5 /= num_queries
    r10 /= num_queries

    print("-" * 40)
    print("EVALUATION RESULTS (Overall)")
    print("-" * 40)
    print(f"MRR:       {mrr:.4f}")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"Queries:   {num_queries}")

    if len(lang_stats) > 1:
        print("-" * 40)
        print("Per-Language Breakdown")
        print("-" * 40)
        for lang, s in sorted(lang_stats.items()):
            n = s["count"]
            print(f"\n  [{lang}]  ({n} queries)")
            print(f"  MRR:       {s['mrr'] / n:.4f}")
            print(f"  Recall@1:  {s['r1'] / n:.4f}")
            print(f"  Recall@5:  {s['r5'] / n:.4f}")
            print(f"  Recall@10: {s['r10'] / n:.4f}")

    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Code Search Model")
    
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the saved model directory (e.g., cs_models/)")
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl", help="Path to evaluation queries")
    parser.add_argument("--distractors_file", type=str, default="data/distractors.jsonl", help="Path to distractors")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--code_len", type=int, default=256, help="Max length for code tokens")
    parser.add_argument("--nl_len", type=int, default=128, help="Max length for natural language tokens")
    
    args = parser.parse_args()
    evaluate(args)
