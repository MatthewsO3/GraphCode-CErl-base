
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from model import GraphCodeBERTDataset, GraphCodeBERTWithEdgePrediction, MLMWithEdgePredictionCollator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PerformanceTracker:
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.history = {
            'epoch': [],
            'train_total_loss': [],
            'train_mlm_loss': [],
            'train_edge_loss': [],
            'train_batch_losses': [],
            'train_mlm_batch_losses': [],
            'train_edge_batch_losses': [],
            'val_total_loss': [],
            'val_mlm_loss': [],
            'val_edge_loss': [],
            'val_batch_losses': [],
            'val_mlm_batch_losses': [],
            'val_edge_batch_losses': [],
            'learning_rate': [],
            'best_val_loss': None,
            'best_epoch': None,
        }

    def log_batch(self, phase: str, total_loss, mlm_loss, edge_loss):
        if phase == 'train':
            self.history['train_batch_losses'].append(total_loss)
            self.history['train_mlm_batch_losses'].append(mlm_loss if mlm_loss else 0)
            self.history['train_edge_batch_losses'].append(edge_loss if edge_loss else 0)
        else:
            self.history['val_batch_losses'].append(total_loss)
            self.history['val_mlm_batch_losses'].append(mlm_loss if mlm_loss else 0)
            self.history['val_edge_batch_losses'].append(edge_loss if edge_loss else 0)

    def log_epoch(self, epoch: int, phase: str, total_loss, mlm_loss, edge_loss, lr=None):
        if phase == 'train':
            self.history['epoch'].append(epoch)
            self.history['train_total_loss'].append(total_loss)
            self.history['train_mlm_loss'].append(mlm_loss)
            self.history['train_edge_loss'].append(edge_loss)
            if lr is not None:
                self.history['learning_rate'].append(lr)
        else:
            self.history['val_total_loss'].append(total_loss)
            self.history['val_mlm_loss'].append(mlm_loss)
            self.history['val_edge_loss'].append(edge_loss)

    def update_best(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def should_stop_early(self) -> bool:
        return self.patience_counter >= self.patience

    def _compute_summary(self) -> Dict:
        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['best_epoch'],
            'best_val_loss': self.history['best_val_loss'],
            'final_train_loss': self.history['train_total_loss'][-1] if self.history['train_total_loss'] else None,
            'final_val_loss': self.history['val_total_loss'][-1] if self.history['val_total_loss'] else None,
            'min_train_loss': min(self.history['train_total_loss']) if self.history['train_total_loss'] else None,
            'min_val_loss': min(self.history['val_total_loss']) if self.history['val_total_loss'] else None,
            'final_train_mlm_loss': self.history['train_mlm_loss'][-1] if self.history['train_mlm_loss'] else None,
            'final_train_edge_loss': self.history['train_edge_loss'][-1] if self.history['train_edge_loss'] else None,
            'final_val_mlm_loss': self.history['val_mlm_loss'][-1] if self.history['val_mlm_loss'] else None,
            'final_val_edge_loss': self.history['val_edge_loss'][-1] if self.history['val_edge_loss'] else None,
            'total_batches_train': len(self.history['train_batch_losses']),
            'total_batches_val': len(self.history['val_batch_losses']),
        }

    def _save_history_json(self):
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")

    def _save_summary_json(self):
        summary = self._compute_summary()
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training summary to {summary_path}")

    def _save_metrics_csv(self):
        try:
            csv_path = self.output_dir / 'training_metrics.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Epoch', 'Train Total Loss', 'Train MLM Loss', 'Train Edge Loss',
                    'Val Total Loss', 'Val MLM Loss', 'Val Edge Loss', 'Learning Rate'
                ])
                for i in range(len(self.history['epoch'])):
                    writer.writerow([
                        self.history['epoch'][i],
                        self.history['train_total_loss'][i],
                        self.history['train_mlm_loss'][i],
                        self.history['train_edge_loss'][i],
                        self.history['val_total_loss'][i] if i < len(self.history['val_total_loss']) else '',
                        self.history['val_mlm_loss'][i] if i < len(self.history['val_mlm_loss']) else '',
                        self.history['val_edge_loss'][i] if i < len(self.history['val_edge_loss']) else '',
                        self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else '',
                    ])
            print(f"Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"Could not save CSV: {e}")

    def save(self):
        self._save_history_json()
        self._save_summary_json()
        self._save_metrics_csv()


class ModelCheckpointManager:
    def __init__(self, output_dir: str, keep_last_n: int = 999):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir = self.output_dir / 'best_model'
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_list = []

    def save_checkpoint(self, model: GraphCodeBERTWithEdgePrediction, tokenizer: RobertaTokenizer, epoch: int):
        checkpoint_dir = self.checkpoints_dir / f'epoch_{epoch:03d}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

        self.checkpoint_list.append(checkpoint_dir)

        if len(self.checkpoint_list) > self.keep_last_n:
            old_checkpoint = self.checkpoint_list.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")

        print(f"Saved checkpoint to {checkpoint_dir}")

    def save_best_model(self, model: GraphCodeBERTWithEdgePrediction, tokenizer: RobertaTokenizer):
        model.save_pretrained(str(self.best_model_dir))
        tokenizer.save_pretrained(str(self.best_model_dir))
        print(f"Saved best model to {self.best_model_dir}")

    def get_best_model_path(self) -> str:
        return str(self.best_model_dir)

    def get_checkpoint_paths(self) -> list:
        return [str(cp) for cp in self.checkpoint_list]


def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        use_amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
        use_amp = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_amp = False

    return device, use_amp


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


def load_config_and_set_defaults(parser):
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            train_config = config_data.get("train", {})
            parser.set_defaults(**train_config)
        print(f"Loaded config from: {config_path}")
        return project_root
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def setup_model_and_data(args, device, project_root):
    print("Loading GraphCodeBERT...")

    # Check if loading from checkpoint (e.g., Erlang trained model)
    checkpoint_path = getattr(args, 'checkpoint_path', None)

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GraphCodeBERTWithEdgePrediction.from_pretrained(checkpoint_path).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    else:
        print("Loading base model...")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(device)

    print("Model loaded successfully")

    data_path = project_root / args.data_file
    print(f"Looking for data file at: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            f"Make sure data_file path in config.json is correct and relative to project root.\n"
            f"Project root: {project_root}"
        )

    print(f"Found data file: {data_path}")
    full_dataset = GraphCodeBERTDataset(str(data_path), tokenizer, args.max_length)

    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        collate_fn=collator, num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )

    return model, tokenizer, train_dl, val_dl


def setup_optimizer_and_scheduler(model, args, train_dl):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs
    )
    return optimizer, scheduler


def print_training_config(args, device, use_amp):
    print("\n--- Training Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  use_amp: {use_amp}")
    print(f"  device: {device}")
    print("------------------------------\n")


def train_epoch(model, dataloader, optimizer, scheduler, device, tracker: PerformanceTracker,
                scaler, use_amp=False) -> Tuple[float, float, float]:
    model.train()
    total_loss = total_mlm = total_edge = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        try:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # ✅ NEW: Model returns dict now
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        position_ids=batch['position_ids'],
                        labels=batch['labels'],
                        edge_batch_idx=batch['edge_batch_idx'],
                        edge_node1_pos=batch['edge_node1_pos'],
                        edge_node2_pos=batch['edge_node2_pos'],
                        edge_labels=batch['edge_labels']
                    )

                    loss = outputs['loss']
                    mlm_loss = outputs['mlm_loss']
                    edge_loss = outputs['edge_loss']

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # ✅ NEW: Model returns dict now
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'],
                    labels=batch['labels'],
                    edge_batch_idx=batch['edge_batch_idx'],
                    edge_node1_pos=batch['edge_node1_pos'],
                    edge_node2_pos=batch['edge_node2_pos'],
                    edge_labels=batch['edge_labels']
                )

                loss = outputs['loss']
                mlm_loss = outputs['mlm_loss']
                edge_loss = outputs['edge_loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            if mlm_loss: total_mlm += mlm_loss.item()
            if edge_loss: total_edge += edge_loss.item()
            batch_count += 1

            tracker.log_batch('train', loss.item(),
                              mlm_loss.item() if mlm_loss else None,
                              edge_loss.item() if edge_loss else None)

            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'mlm': f'{mlm_loss.item() if mlm_loss else 0:.4f}',
                'edge': f'{edge_loss.item() if edge_loss else 0:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nOUT OF MEMORY ERROR!")
                print(f"   Batch size: {batch['input_ids'].shape[0]}")
                print(f"   Sequence length: {batch['input_ids'].shape[1]}")
                print(f"   Try reducing batch_size or max_length")
                raise
            else:
                raise

        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


def validate(model, dataloader, device, tracker: PerformanceTracker, use_amp=False) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total_mlm = total_edge = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # ✅ NEW: Model returns dict now
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        position_ids=batch['position_ids'],
                        labels=batch['labels'],
                        edge_batch_idx=batch['edge_batch_idx'],
                        edge_node1_pos=batch['edge_node1_pos'],
                        edge_node2_pos=batch['edge_node2_pos'],
                        edge_labels=batch['edge_labels']
                    )
            else:
                # ✅ NEW: Model returns dict now
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'],
                    labels=batch['labels'],
                    edge_batch_idx=batch['edge_batch_idx'],
                    edge_node1_pos=batch['edge_node1_pos'],
                    edge_node2_pos=batch['edge_node2_pos'],
                    edge_labels=batch['edge_labels']
                )

            loss = outputs['loss']
            mlm_loss = outputs['mlm_loss']
            edge_loss = outputs['edge_loss']

            total_loss += loss.item()
            if mlm_loss: total_mlm += mlm_loss.item()
            if edge_loss: total_edge += edge_loss.item()
            batch_count += 1

            tracker.log_batch('val', loss.item(),
                              mlm_loss.item() if mlm_loss else None,
                              edge_loss.item() if edge_loss else None)

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'mlm': f'{mlm_loss.item() if mlm_loss else 0:.4f}',
                'edge': f'{edge_loss.item() if edge_loss else 0:.4f}'
            })

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


def print_epoch_results(epoch, args, train_loss, train_mlm, train_edge, val_loss, val_mlm, val_edge,
                        current_lr, tracker):
    print(f"\n{'─' * 70}")
    print(f"Epoch {epoch + 1} Results:")
    print(f"  Train - Total: {train_loss:.6f}, MLM: {train_mlm:.6f}, Edge: {train_edge:.6f}")
    print(f"  Val   - Total: {val_loss:.6f}, MLM: {val_mlm:.6f}, Edge: {val_edge:.6f}")
    print(f"  Learning Rate: {current_lr:.6e}")
    print(
        f"  Best Val Loss: {tracker.best_val_loss:.6f} (Epoch {tracker.history['best_epoch'] + 1 if tracker.history['best_epoch'] is not None else 'N/A'})")
    print(f"  Patience:      {tracker.patience_counter}/{args.early_stopping_patience}")
    print(f"{'─' * 70}")


def clear_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def training_loop(model, train_dl, val_dl, optimizer, scheduler, device, args, tracker, checkpoint_manager, scaler,
                  use_amp):
    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        clear_cache(device)

        train_loss, train_mlm, train_edge = train_epoch(model, train_dl, optimizer, scheduler, device, tracker, scaler,
                                                        use_amp)

        clear_cache(device)

        val_loss, val_mlm, val_edge = validate(model, val_dl, device, tracker, use_amp)

        current_lr = optimizer.param_groups[0]['lr']

        tracker.log_epoch(epoch, 'train', train_loss, train_mlm, train_edge, current_lr)
        tracker.log_epoch(epoch, 'val', val_loss, val_mlm, val_edge)

        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"\n  Peak GPU Memory: {peak_mem:.2f} GB")

        print_epoch_results(epoch, args, train_loss, train_mlm, train_edge, val_loss, val_mlm, val_edge,
                            current_lr, tracker)

        checkpoint_manager.save_checkpoint(model, args.tokenizer, epoch)

        if tracker.update_best(val_loss, epoch):
            print(f"\nNew best model! Saving best model...")
            checkpoint_manager.save_best_model(model, args.tokenizer)
        else:
            print(f"\nNo improvement. Patience: {tracker.patience_counter}/{args.early_stopping_patience}")

        if tracker.should_stop_early():
            print(f"\nEarly stopping triggered!")
            print(f"   No improvement for {args.early_stopping_patience} epochs")
            print(f"   Best loss: {tracker.best_val_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
            break

    print(f"\n{'=' * 70}")
    print(f"Training completed!")
    print(f"Best val loss: {tracker.best_val_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train GraphCodeBERT with Edge Prediction')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to training data file (relative to project root)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (relative to project root)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to load (e.g., Erlang trained model)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=None, help='Max sequence length')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
    parser.add_argument('--mlm_probability', type=float, default=None, help='MLM masking probability')
    parser.add_argument('--validation_split', type=float, default=None, help='Validation split ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--use_amp', default=True, action='store_true', help='Use mixed precision training')

    project_root = load_config_and_set_defaults(parser)
    args = parser.parse_args()

    if not args.data_file:
        parser.error("data_file must be specified in config.json or via --data_file argument.")
    if not args.output_dir:
        parser.error("output_dir must be specified in config.json or via --output_dir argument.")

    set_seed(42)

    device, use_amp = setup_device()
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = PerformanceTracker(str(output_path), patience=args.early_stopping_patience)
    checkpoint_manager = ModelCheckpointManager(str(output_path))

    model, tokenizer, train_dl, val_dl = setup_model_and_data(args, device, project_root)
    args.tokenizer = tokenizer

    optimizer, scheduler = setup_optimizer_and_scheduler(model, args, train_dl)
    scaler = GradScaler() if use_amp else None

    print_training_config(args, device, use_amp)

    training_loop(model, train_dl, val_dl, optimizer, scheduler, device, args, tracker, checkpoint_manager, scaler,
                  use_amp)

    print("=" * 70)
    print("SAVING PERFORMANCE METRICS...")
    print("=" * 70)
    tracker.save()
    print(f"\nAll results saved to {output_path}")
    print(f"Best model saved at: {checkpoint_manager.get_best_model_path()}")
    print(f"Checkpoints saved at: {checkpoint_manager.checkpoints_dir}")


if __name__ == "__main__":
    main()