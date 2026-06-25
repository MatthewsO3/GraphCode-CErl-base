import argparse
import torch
import json
import csv
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import random
import shutil
from typing import Dict, List, Optional
from model import CodeSearchModel


def collate(batch):
    return tuple(torch.stack(x) for x in zip(*batch))


class CodeSearchDataset(Dataset):
    def __init__(self, tokenizer, file_path, code_len, nl_len):
        self.examples = []
        self.tokenizer = tokenizer
        self.code_len = code_len
        self.nl_len = nl_len

        with open(file_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def encode(self, text, max_len):
        return self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def __getitem__(self, i):
        e = self.examples[i]

        def proc(text, max_len):
            enc = self.encode(text, max_len)
            return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

        return (
            *proc(e['code'], self.code_len),
            *proc(e['good_docstring'], self.nl_len),
            *proc(e['bad1_docstring'], self.nl_len),
            *proc(e['bad2_docstring'], self.nl_len),
        )


class PerformanceTracker:
    """Records per-batch and per-epoch training metrics and persists them to disk."""

    def __init__(self, output_dir: str, patience: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = float("inf")

        self.history: Dict = {
            "epoch": [],
            "train_total_loss": [],
            "train_ce_loss": [],
            "train_neg_loss": [],
            "train_batch_losses": [],
            "train_ce_batch_losses": [],
            "train_neg_batch_losses": [],
            "learning_rate": [],
            "best_loss": None,
            "best_epoch": None,
        }

    def log_batch(self, total_loss: float, ce_loss: float, neg_loss: float) -> None:
        self.history["train_batch_losses"].append(total_loss)
        self.history["train_ce_batch_losses"].append(ce_loss)
        self.history["train_neg_batch_losses"].append(neg_loss)

    def log_epoch(self, epoch: int, total_loss: float, ce_loss: float, neg_loss: float, lr: float) -> None:
        self.history["epoch"].append(epoch)
        self.history["train_total_loss"].append(total_loss)
        self.history["train_ce_loss"].append(ce_loss)
        self.history["train_neg_loss"].append(neg_loss)
        self.history["learning_rate"].append(lr)

    def update_best(self, loss: float, epoch: int) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.history["best_loss"] = loss
            self.history["best_epoch"] = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def should_stop_early(self) -> bool:
        return self.patience_counter >= self.patience

    def _compute_summary(self) -> Dict:
        return {
            "total_epochs": len(self.history["epoch"]),
            "best_epoch": self.history["best_epoch"],
            "best_loss": self.history["best_loss"],
            "final_train_loss": self.history["train_total_loss"][-1] if self.history["train_total_loss"] else None,
            "min_train_loss": min(self.history["train_total_loss"]) if self.history["train_total_loss"] else None,
            "final_ce_loss": self.history["train_ce_loss"][-1] if self.history["train_ce_loss"] else None,
            "final_neg_loss": self.history["train_neg_loss"][-1] if self.history["train_neg_loss"] else None,
            "total_batches": len(self.history["train_batch_losses"]),
        }

    def save(self) -> None:
        # Full history JSON
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")

        # Summary JSON
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._compute_summary(), f, indent=2)
        print(f"Saved training summary to {summary_path}")

        # CSV
        try:
            csv_path = self.output_dir / "training_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Total Loss", "CE Loss", "Neg Loss", "Learning Rate"])
                for i in range(len(self.history["epoch"])):
                    writer.writerow([
                        self.history["epoch"][i],
                        self.history["train_total_loss"][i],
                        self.history["train_ce_loss"][i],
                        self.history["train_neg_loss"][i],
                        self.history["learning_rate"][i] if i < len(self.history["learning_rate"]) else "",
                    ])
            print(f"Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"Could not save CSV: {e}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, tokenizer, output_dir, epoch, best=False):
    base = Path(output_dir)

    epoch_dir = base / "checkpoints" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(epoch_dir))
    tokenizer.save_pretrained(str(epoch_dir))
    print(f"Saved checkpoint to {epoch_dir}")

    if best:
        best_dir = base / "best_model"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(epoch_dir, best_dir)
        print(f"Saved best model to {best_dir}")


def print_epoch_results(epoch, args, total_loss, ce_loss, neg_loss, lr, tracker):
    print(f"\n{'─' * 70}")
    print(f"Epoch {epoch + 1} Results:")
    print(f"  Total Loss: {total_loss:.6f}  |  CE Loss: {ce_loss:.6f}  |  Neg Loss: {neg_loss:.6f}")
    print(f"  Learning Rate: {lr:.6e}")
    print(f"  Best Loss: {tracker.best_loss:.6f} "
          f"(Epoch {tracker.history['best_epoch'] + 1 if tracker.history['best_epoch'] is not None else 'N/A'})")
    print(f"  Patience: {tracker.patience_counter}/{args.early_stopping_patience}")
    print(f"{'─' * 70}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model = CodeSearchModel(base_model).to(device)

    dataset = CodeSearchDataset(tokenizer, args.train_file, args.code_len, args.nl_len)
    loader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=args.batch_size,
        collate_fn=collate
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tracker = PerformanceTracker(str(output_path), patience=args.early_stopping_patience)

    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        model.train()
        total_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(loader, desc="Training")
        for batch in progress_bar:
            batch = [x.to(device) for x in batch]
            code_ids, code_mask, good_ids, good_mask, bad1_ids, bad1_mask, bad2_ids, bad2_mask = batch

            code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
            good_vec = model(nl_inputs=good_ids, attention_mask=good_mask)
            bad1_vec = model(nl_inputs=bad1_ids, attention_mask=bad1_mask)
            bad2_vec = model(nl_inputs=bad2_ids, attention_mask=bad2_mask)

            scores     = torch.einsum("ab,cb->ac", good_vec, code_vec) / args.temp
            bad1_scores = torch.einsum("ab,cb->ac", bad1_vec, code_vec) / args.temp
            bad2_scores = torch.einsum("ab,cb->ac", bad2_vec, code_vec) / args.temp

            augmented_scores = torch.cat([scores, bad1_scores, bad2_scores], dim=1)
            labels = torch.arange(code_ids.size(0), device=device)
            loss = torch.nn.CrossEntropyLoss()(augmented_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batch_count += 1
            current_lr = optimizer.param_groups[0]["lr"]

            tracker.log_batch(loss.item(), loss.item(), 0.0)

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / batch_count:.4f}",
                "lr": f"{current_lr:.2e}",
            })

        epoch_loss = total_loss / batch_count
        current_lr = optimizer.param_groups[0]["lr"]

        tracker.log_epoch(epoch, epoch_loss, epoch_loss, 0.0, current_lr)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n  Peak GPU Memory: {peak_mem:.2f} GB")

        is_best = tracker.update_best(epoch_loss, epoch)
        save_checkpoint(model, tokenizer, args.output_dir, epoch, best=is_best)
        print_epoch_results(epoch, args, epoch_loss, epoch_loss, 0.0, current_lr, tracker)

        if is_best:
            print("New best model!")
        else:
            print(f"No improvement. Patience: {tracker.patience_counter}/{args.early_stopping_patience}")

        if tracker.should_stop_early():
            print(f"\nEarly stopping triggered after {args.early_stopping_patience} epochs without improvement.")
            print(f"Best loss: {tracker.best_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
            break

    print(f"\n{'=' * 70}")
    print(f"Training complete. Best loss: {tracker.best_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
    print(f"{'=' * 70}\n")

    tracker.save()
    print(f"Best model: {output_path / 'best_model'}")
    print(f"All checkpoints: {output_path / 'checkpoints'}")


def load_config(parser):
    parser.add_argument("--config", type=str, default="config.json")
    pre_args, _ = parser.parse_known_args()
    if Path(pre_args.config).exists():
        with open(pre_args.config) as f:
            config = json.load(f)
        train_config = config.get("train", {})
        parser.set_defaults(**train_config)
        print(f"Loaded config from {pre_args.config}")
    else:
        print(f"No config file found at {pre_args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file")
    parser.add_argument("--model_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--code_len", type=int, default=256)
    parser.add_argument("--nl_len", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--neg_weight", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    load_config(parser)
    args = parser.parse_args()

    set_seed(42)
    train(args)