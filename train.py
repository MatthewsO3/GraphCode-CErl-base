import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from model import (
    GraphCodeBERTDataset,
    GraphCodeBERTWithEdgePrediction,
    MLMWithEdgePredictionCollator,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    When CUDA is available, seeds all GPU devices as well.

    :param seed: Integer seed value to apply across all RNG backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PerformanceTracker:
    """Records per-batch and per-epoch training metrics and persists them to disk.

    Maintains a history dictionary covering train and validation losses
    (total, MLM, and edge components) and the learning rate at each epoch.
    Also implements patience-based early stopping by tracking whether the
    validation loss has improved relative to the best seen so far.

    Output files written by :meth:`save`:

    * ``training_history.json`` — full per-epoch and per-batch history.
    * ``training_summary.json`` — aggregated statistics (best epoch, min
      losses, final losses, batch counts).
    * ``training_metrics.csv`` — one row per epoch for quick inspection.

    :param output_dir: Directory where all metric files will be written.
        Created automatically if it does not exist.
    :param patience: Number of consecutive epochs without validation loss
        improvement before :meth:`should_stop_early` returns ``True``.
    """

    def __init__(self, output_dir: str, patience: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float("inf")

        self.history: Dict = {
            "epoch": [],
            "train_total_loss": [],
            "train_mlm_loss": [],
            "train_edge_loss": [],
            "train_batch_losses": [],
            "train_mlm_batch_losses": [],
            "train_edge_batch_losses": [],
            "val_total_loss": [],
            "val_mlm_loss": [],
            "val_edge_loss": [],
            "val_batch_losses": [],
            "val_mlm_batch_losses": [],
            "val_edge_batch_losses": [],
            "learning_rate": [],
            "best_val_loss": None,
            "best_epoch": None,
        }

    def log_batch(
        self,
        phase: str,
        total_loss: float,
        mlm_loss: Optional[float],
        edge_loss: Optional[float],
    ) -> None:
        """Append per-batch losses to the appropriate history lists.

        :param phase: Either ``"train"`` or ``"val"``.
        :param total_loss: Combined scalar loss for this batch.
        :param mlm_loss: MLM loss component, or ``None`` if not computed.
        :param edge_loss: Edge-prediction loss component, or ``None`` if not
            computed.
        """
        if phase == "train":
            self.history["train_batch_losses"].append(total_loss)
            self.history["train_mlm_batch_losses"].append(mlm_loss if mlm_loss else 0)
            self.history["train_edge_batch_losses"].append(
                edge_loss if edge_loss else 0
            )
        else:
            self.history["val_batch_losses"].append(total_loss)
            self.history["val_mlm_batch_losses"].append(mlm_loss if mlm_loss else 0)
            self.history["val_edge_batch_losses"].append(edge_loss if edge_loss else 0)

    def log_epoch(
        self,
        epoch: int,
        phase: str,
        total_loss: float,
        mlm_loss: float,
        edge_loss: float,
        lr: Optional[float] = None,
    ) -> None:
        """Append per-epoch aggregated losses (and optionally the LR) to history.

        For the ``"train"`` phase, also records the epoch index and, if
        supplied, the current learning rate.  For ``"val"``, only the loss
        components are appended.

        :param epoch: Zero-based epoch index.
        :param phase: Either ``"train"`` or ``"val"``.
        :param total_loss: Mean total loss over all batches in the epoch.
        :param mlm_loss: Mean MLM loss over all batches in the epoch.
        :param edge_loss: Mean edge-prediction loss over all batches in the
            epoch.
        :param lr: Current learning rate at the end of the epoch.  Only
            recorded for the ``"train"`` phase; ignored for ``"val"``.
        """
        if phase == "train":
            self.history["epoch"].append(epoch)
            self.history["train_total_loss"].append(total_loss)
            self.history["train_mlm_loss"].append(mlm_loss)
            self.history["train_edge_loss"].append(edge_loss)
            if lr is not None:
                self.history["learning_rate"].append(lr)
        else:
            self.history["val_total_loss"].append(total_loss)
            self.history["val_mlm_loss"].append(mlm_loss)
            self.history["val_edge_loss"].append(edge_loss)

    def update_best(self, val_loss: float, epoch: int) -> bool:
        """Compare ``val_loss`` against the best seen so far and update state.

        When a new best is found, resets the patience counter and records the
        best loss and epoch in :attr:`history`.  Otherwise, increments the
        patience counter.

        :param val_loss: Validation loss for the completed epoch.
        :param epoch: Zero-based epoch index, stored when a new best is found.
        :returns: ``True`` if ``val_loss`` is a new best, ``False`` otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history["best_val_loss"] = val_loss
            self.history["best_epoch"] = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def should_stop_early(self) -> bool:
        """Return whether the patience limit has been reached.

        :returns: ``True`` when the number of consecutive epochs without
            improvement equals or exceeds :attr:`patience`.
        """
        return self.patience_counter >= self.patience

    def _compute_summary(self) -> Dict:
        """Build a summary dictionary of aggregate training statistics.

        :returns: A flat dictionary containing total epochs, best epoch and
            loss, final and minimum train/val losses, final MLM and edge losses,
            and total batch counts for train and validation phases.
        """
        return {
            "total_epochs": len(self.history["epoch"]),
            "best_epoch": self.history["best_epoch"],
            "best_val_loss": self.history["best_val_loss"],
            "final_train_loss": (
                self.history["train_total_loss"][-1]
                if self.history["train_total_loss"]
                else None
            ),
            "final_val_loss": (
                self.history["val_total_loss"][-1]
                if self.history["val_total_loss"]
                else None
            ),
            "min_train_loss": (
                min(self.history["train_total_loss"])
                if self.history["train_total_loss"]
                else None
            ),
            "min_val_loss": (
                min(self.history["val_total_loss"])
                if self.history["val_total_loss"]
                else None
            ),
            "final_train_mlm_loss": (
                self.history["train_mlm_loss"][-1]
                if self.history["train_mlm_loss"]
                else None
            ),
            "final_train_edge_loss": (
                self.history["train_edge_loss"][-1]
                if self.history["train_edge_loss"]
                else None
            ),
            "final_val_mlm_loss": (
                self.history["val_mlm_loss"][-1]
                if self.history["val_mlm_loss"]
                else None
            ),
            "final_val_edge_loss": (
                self.history["val_edge_loss"][-1]
                if self.history["val_edge_loss"]
                else None
            ),
            "total_batches_train": len(self.history["train_batch_losses"]),
            "total_batches_val": len(self.history["val_batch_losses"]),
        }

    def _save_history_json(self) -> None:
        """Write the full history dictionary to ``training_history.json``."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")

    def _save_summary_json(self) -> None:
        """Write the aggregated summary to ``training_summary.json``."""
        summary = self._compute_summary()
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training summary to {summary_path}")

    def _save_metrics_csv(self) -> None:
        """Write per-epoch metrics to ``training_metrics.csv``.

        Each row contains the epoch index, train and validation losses (total,
        MLM, edge), and the learning rate.  Missing validation entries for
        partially completed runs are written as empty strings.  Any exception
        during writing is caught and reported without raising.
        """
        try:
            csv_path = self.output_dir / "training_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Epoch",
                        "Train Total Loss",
                        "Train MLM Loss",
                        "Train Edge Loss",
                        "Val Total Loss",
                        "Val MLM Loss",
                        "Val Edge Loss",
                        "Learning Rate",
                    ]
                )
                for i in range(len(self.history["epoch"])):
                    writer.writerow(
                        [
                            self.history["epoch"][i],
                            self.history["train_total_loss"][i],
                            self.history["train_mlm_loss"][i],
                            self.history["train_edge_loss"][i],
                            (
                                self.history["val_total_loss"][i]
                                if i < len(self.history["val_total_loss"])
                                else ""
                            ),
                            (
                                self.history["val_mlm_loss"][i]
                                if i < len(self.history["val_mlm_loss"])
                                else ""
                            ),
                            (
                                self.history["val_edge_loss"][i]
                                if i < len(self.history["val_edge_loss"])
                                else ""
                            ),
                            (
                                self.history["learning_rate"][i]
                                if i < len(self.history["learning_rate"])
                                else ""
                            ),
                        ]
                    )
            print(f"Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"Could not save CSV: {e}")

    def save(self) -> None:
        """Persist all metric files to :attr:`output_dir`.

        Calls :meth:`_save_history_json`, :meth:`_save_summary_json`, and
        :meth:`_save_metrics_csv` in sequence.
        """
        self._save_history_json()
        self._save_summary_json()
        self._save_metrics_csv()


class ModelCheckpointManager:
    """Saves per-epoch checkpoints and tracks the best-performing model.

    Checkpoints are written to ``<output_dir>/checkpoints/epoch_NNN/`` and
    the best model is kept separately at ``<output_dir>/best_model/``.  When
    more than ``keep_last_n`` checkpoints have accumulated, the oldest is
    deleted automatically.

    :param output_dir: Root directory under which ``checkpoints/`` and
        ``best_model/`` subdirectories are created.
    :param keep_last_n: Maximum number of epoch checkpoints to retain on disk
        at any one time.  Defaults to ``999`` (effectively unlimited).
    """

    def __init__(self, output_dir: str, keep_last_n: int = 999) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir = self.output_dir / "best_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_list: List[Path] = []

    def save_checkpoint(
        self,
        model: GraphCodeBERTWithEdgePrediction,
        tokenizer: RobertaTokenizer,
        epoch: int,
    ) -> None:
        """Save a model and tokenizer checkpoint for the given epoch.

        The checkpoint is written to
        ``<checkpoints_dir>/epoch_<epoch:03d>/``.  If the total number of
        retained checkpoints exceeds :attr:`keep_last_n`, the oldest
        checkpoint directory is deleted from disk.

        :param model: The model whose weights will be saved via
            ``save_pretrained``.
        :param tokenizer: The tokenizer to co-locate with the checkpoint.
        :param epoch: Zero-based epoch index used to name the checkpoint
            directory.
        """
        checkpoint_dir = self.checkpoints_dir / f"epoch_{epoch:03d}"
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

    def save_best_model(
        self,
        model: GraphCodeBERTWithEdgePrediction,
        tokenizer: RobertaTokenizer,
    ) -> None:
        """Overwrite the best-model directory with the current model state.

        :param model: The model to save via ``save_pretrained``.
        :param tokenizer: The tokenizer to co-locate with the best model.
        """
        model.save_pretrained(str(self.best_model_dir))
        tokenizer.save_pretrained(str(self.best_model_dir))
        print(f"Saved best model to {self.best_model_dir}")

    def get_best_model_path(self) -> str:
        """Return the absolute path to the best-model directory as a string.

        :returns: String path to ``<output_dir>/best_model/``.
        """
        return str(self.best_model_dir)

    def get_checkpoint_paths(self) -> List[str]:
        """Return a list of paths for all currently retained checkpoints.

        :returns: List of string paths, one per retained checkpoint directory,
            in chronological order.
        """
        return [str(cp) for cp in self.checkpoint_list]


def setup_device() -> Tuple[torch.device, bool]:
    """Select the best available compute device and whether to use AMP.

    Priority order: Apple Silicon MPS → NVIDIA CUDA → CPU.  AMP
    (automatic mixed precision) is enabled only for CUDA, as it is not
    supported on MPS or CPU.

    :returns: A 2-tuple ``(device, use_amp)`` where ``device`` is the
        selected :class:`torch.device` and ``use_amp`` is ``True`` only
        when CUDA is available.
    """
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


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Locate the project root by searching upward for ``config.json``.

    :param start_path: Directory from which to begin the upward search.
        Defaults to the directory containing this source file.
    :returns: The nearest ancestor directory that contains ``config.json``.
    :raises FileNotFoundError: If no ``config.json`` is found before reaching
        the filesystem root.
    """
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current = start_path
    while True:
        config_path = current / "config.json"
        if config_path.exists():
            return current

        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find project root. "
                "Make sure config.json exists in the project root directory."
            )
        current = parent


def load_config_and_set_defaults(parser: argparse.ArgumentParser) -> Path:
    """Load ``config.json`` and apply its ``train`` section as argument defaults.

    Locates the project root via :func:`find_project_root`, reads
    ``config.json``, and calls :meth:`~argparse.ArgumentParser.set_defaults`
    with all keys found under the ``"train"`` key.  This allows CLI arguments
    to override config values while still providing sensible defaults without
    repeating them in the argument definitions.

    :param parser: The :class:`~argparse.ArgumentParser` instance whose
        defaults will be updated in-place.
    :returns: The resolved project root :class:`~pathlib.Path`, which callers
        use to resolve relative data and output paths from the config.
    :raises FileNotFoundError: If the project root or ``config.json`` cannot
        be found.
    """
    project_root = find_project_root()
    config_path = project_root / "config.json"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            train_config = config_data.get("train", {})
            parser.set_defaults(**train_config)
        print(f"Loaded config from: {config_path}")
        return project_root
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def setup_model_and_data(
    args: argparse.Namespace,
    device: torch.device,
    project_root: Path,
) -> Tuple[GraphCodeBERTWithEdgePrediction, RobertaTokenizer, DataLoader, DataLoader]:
    """Initialise the model, tokenizer, and train/validation data loaders.

    When ``args.checkpoint_path`` is set, the model and tokenizer are loaded
    from that checkpoint directory; otherwise the base
    ``microsoft/graphcodebert-base`` weights are used.

    The dataset is loaded from ``project_root / args.data_file``, split into
    train and validation subsets according to ``args.validation_split``, and
    wrapped in :class:`~torch.utils.data.DataLoader` instances using
    :class:`MLMWithEdgePredictionCollator`.

    :param args: Parsed argument namespace providing ``checkpoint_path``,
        ``data_file``, ``max_length``, ``validation_split``, ``batch_size``,
        and ``mlm_probability``.
    :param device: Target device; the model is moved to this device before
        being returned.
    :param project_root: Absolute path to the project root, used to resolve
        ``args.data_file``.
    :returns: A 4-tuple ``(model, tokenizer, train_dataloader,
        val_dataloader)``.
    :raises FileNotFoundError: If the data file resolved from
        ``project_root / args.data_file`` does not exist.
    """
    print("Loading GraphCodeBERT...")

    checkpoint_path = getattr(args, "checkpoint_path", None)

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GraphCodeBERTWithEdgePrediction.from_pretrained(checkpoint_path).to(
            device
        )
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    else:
        print("Loading base model...")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(
            device
        )

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

    collator = MLMWithEdgePredictionCollator(
        tokenizer, mlm_probability=args.mlm_probability
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    return model, tokenizer, train_dl, val_dl


def setup_optimizer_and_scheduler(
    model: GraphCodeBERTWithEdgePrediction,
    args: argparse.Namespace,
    train_dl: DataLoader,
) -> Tuple[AdamW, object]:
    """Create an AdamW optimiser and a linear warm-up LR scheduler.

    The total number of training steps is calculated as
    ``len(train_dl) * args.epochs``.

    :param model: The model whose parameters will be optimised.
    :param args: Parsed argument namespace providing ``learning_rate``,
        ``weight_decay``, ``warmup_steps``, and ``epochs``.
    :param train_dl: Training data loader, used to determine the total step
        count for the scheduler.
    :returns: A 2-tuple ``(optimizer, scheduler)``.
    """
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs,
    )
    return optimizer, scheduler


def print_training_config(
    args: argparse.Namespace, device: torch.device, use_amp: bool
) -> None:
    """Print a formatted summary of the training configuration to stdout.

    :param args: Parsed argument namespace; all key-value pairs are printed.
    :param device: The selected compute device.
    :param use_amp: Whether automatic mixed precision is enabled.
    """
    print("\n--- Training Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  use_amp: {use_amp}")
    print(f"  device: {device}")
    print("------------------------------\n")


def train_epoch(
    model: GraphCodeBERTWithEdgePrediction,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    tracker: PerformanceTracker,
    scaler: Optional[GradScaler],
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """Run one full training epoch over ``dataloader``.

    For each batch, moves tensors to ``device``, runs a forward pass
    (optionally under AMP autocast), backpropagates, clips gradients to
    norm 1.0, and steps the optimiser and scheduler.  Per-batch losses are
    forwarded to ``tracker`` and displayed on a tqdm progress bar.

    Raises informative messages for CUDA out-of-memory errors before
    re-raising.  GPU/MPS cache is cleared in the ``finally`` block after
    every batch.

    :param model: The model to train.
    :param dataloader: Training data loader yielding batches from
        :class:`MLMWithEdgePredictionCollator`.
    :param optimizer: The optimiser to step after each batch.
    :param scheduler: A Hugging Face LR scheduler stepped after the optimiser.
    :param device: Compute device to which batch tensors are moved.
    :param tracker: :class:`PerformanceTracker` that receives per-batch loss
        logs.
    :param scaler: :class:`~torch.cuda.amp.GradScaler` used when
        ``use_amp`` is ``True``; may be ``None`` otherwise.
    :param use_amp: If ``True``, wraps the forward pass in
        ``torch.amp.autocast`` and uses ``scaler`` for gradient scaling.
    :returns: A 3-tuple ``(mean_total_loss, mean_mlm_loss, mean_edge_loss)``
        averaged over all batches in the epoch.
    :raises RuntimeError: Re-raised after printing diagnostic information when
        a CUDA out-of-memory error is detected.
    """
    model.train()
    total_loss = total_mlm = total_edge = 0.0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        try:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if use_amp:
                with torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu"
                ):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        labels=batch["labels"],
                        edge_batch_idx=batch["edge_batch_idx"],
                        edge_node1_pos=batch["edge_node1_pos"],
                        edge_node2_pos=batch["edge_node2_pos"],
                        edge_labels=batch["edge_labels"],
                    )
                    loss = outputs["loss"]
                    mlm_loss = outputs["mlm_loss"]
                    edge_loss = outputs["edge_loss"]

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"],
                    edge_batch_idx=batch["edge_batch_idx"],
                    edge_node1_pos=batch["edge_node1_pos"],
                    edge_node2_pos=batch["edge_node2_pos"],
                    edge_labels=batch["edge_labels"],
                )
                loss = outputs["loss"]
                mlm_loss = outputs["mlm_loss"]
                edge_loss = outputs["edge_loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            if mlm_loss:
                total_mlm += mlm_loss.item()
            if edge_loss:
                total_edge += edge_loss.item()
            batch_count += 1

            tracker.log_batch(
                "train",
                loss.item(),
                mlm_loss.item() if mlm_loss else None,
                edge_loss.item() if edge_loss else None,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "mlm": f"{mlm_loss.item() if mlm_loss else 0:.4f}",
                    "edge": f"{edge_loss.item() if edge_loss else 0:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nOUT OF MEMORY ERROR!")
                print(f"   Batch size: {batch['input_ids'].shape[0]}")
                print(f"   Sequence length: {batch['input_ids'].shape[1]}")
                print(f"   Try reducing batch_size or max_length")
                raise
            else:
                raise

        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


def validate(
    model: GraphCodeBERTWithEdgePrediction,
    dataloader: DataLoader,
    device: torch.device,
    tracker: PerformanceTracker,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """Run one full validation pass over ``dataloader`` without gradient updates.

    Operates under ``torch.no_grad()``.  Optionally wraps the forward pass in
    AMP autocast.  Per-batch losses are forwarded to ``tracker`` and displayed
    on a tqdm progress bar.

    :param model: The model to evaluate (set to ``eval`` mode internally).
    :param dataloader: Validation data loader.
    :param device: Compute device to which batch tensors are moved.
    :param tracker: :class:`PerformanceTracker` that receives per-batch loss
        logs tagged as ``"val"``.
    :param use_amp: If ``True``, wraps the forward pass in
        ``torch.amp.autocast``.
    :returns: A 3-tuple ``(mean_total_loss, mean_mlm_loss, mean_edge_loss)``
        averaged over all batches.
    """
    model.eval()
    total_loss = total_mlm = total_edge = 0.0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if use_amp:
                with torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu"
                ):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        labels=batch["labels"],
                        edge_batch_idx=batch["edge_batch_idx"],
                        edge_node1_pos=batch["edge_node1_pos"],
                        edge_node2_pos=batch["edge_node2_pos"],
                        edge_labels=batch["edge_labels"],
                    )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"],
                    edge_batch_idx=batch["edge_batch_idx"],
                    edge_node1_pos=batch["edge_node1_pos"],
                    edge_node2_pos=batch["edge_node2_pos"],
                    edge_labels=batch["edge_labels"],
                )

            loss = outputs["loss"]
            mlm_loss = outputs["mlm_loss"]
            edge_loss = outputs["edge_loss"]

            total_loss += loss.item()
            if mlm_loss:
                total_mlm += mlm_loss.item()
            if edge_loss:
                total_edge += edge_loss.item()
            batch_count += 1

            tracker.log_batch(
                "val",
                loss.item(),
                mlm_loss.item() if mlm_loss else None,
                edge_loss.item() if edge_loss else None,
            )

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "mlm": f"{mlm_loss.item() if mlm_loss else 0:.4f}",
                    "edge": f"{edge_loss.item() if edge_loss else 0:.4f}",
                }
            )

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


def print_epoch_results(
    epoch: int,
    args: argparse.Namespace,
    train_loss: float,
    train_mlm: float,
    train_edge: float,
    val_loss: float,
    val_mlm: float,
    val_edge: float,
    current_lr: float,
    tracker: PerformanceTracker,
) -> None:
    """Print a formatted per-epoch results table to stdout.

    :param epoch: Zero-based epoch index.
    :param args: Parsed argument namespace providing
        ``early_stopping_patience``.
    :param train_loss: Mean training total loss for the epoch.
    :param train_mlm: Mean training MLM loss for the epoch.
    :param train_edge: Mean training edge-prediction loss for the epoch.
    :param val_loss: Mean validation total loss for the epoch.
    :param val_mlm: Mean validation MLM loss for the epoch.
    :param val_edge: Mean validation edge-prediction loss for the epoch.
    :param current_lr: Learning rate at the end of the epoch.
    :param tracker: :class:`PerformanceTracker` used to read the current best
        loss, best epoch, and patience counter for display.
    """
    print(f"\n{'─' * 70}")
    print(f"Epoch {epoch + 1} Results:")
    print(
        f"  Train - Total: {train_loss:.6f}, MLM: {train_mlm:.6f}, Edge: {train_edge:.6f}"
    )
    print(f"  Val   - Total: {val_loss:.6f}, MLM: {val_mlm:.6f}, Edge: {val_edge:.6f}")
    print(f"  Learning Rate: {current_lr:.6e}")
    print(
        f"  Best Val Loss: {tracker.best_val_loss:.6f} "
        f"(Epoch {tracker.history['best_epoch'] + 1 if tracker.history['best_epoch'] is not None else 'N/A'})"
    )
    print(f"  Patience:      {tracker.patience_counter}/{args.early_stopping_patience}")
    print(f"{'─' * 70}")


def clear_cache(device: torch.device) -> None:
    """Free unused memory from the GPU cache for the given device.

    A no-op when ``device`` is CPU.

    :param device: The active compute device.  Cache is cleared for
        ``"cuda"`` and ``"mps"`` device types only.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def training_loop(
    model: GraphCodeBERTWithEdgePrediction,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    args: argparse.Namespace,
    tracker: PerformanceTracker,
    checkpoint_manager: ModelCheckpointManager,
    scaler: Optional[GradScaler],
    use_amp: bool,
) -> None:
    """Execute the full multi-epoch training and validation loop.

    For each epoch:

    1. Clears the device cache.
    2. Calls :func:`train_epoch` and :func:`validate`.
    3. Logs epoch metrics to ``tracker`` and prints results via
       :func:`print_epoch_results`.
    4. Reports peak GPU memory usage when running on CUDA.
    5. Saves a per-epoch checkpoint via ``checkpoint_manager``.
    6. Saves the best model when validation loss improves.
    7. Triggers early stopping if ``tracker.should_stop_early()`` returns
       ``True``.

    :param model: The model to train.
    :param train_dl: Training data loader.
    :param val_dl: Validation data loader.
    :param optimizer: The optimiser.
    :param scheduler: The LR scheduler.
    :param device: Compute device.
    :param args: Parsed argument namespace providing ``epochs`` and
        ``early_stopping_patience``.
    :param tracker: :class:`PerformanceTracker` for logging and early-stopping
        decisions.
    :param checkpoint_manager: :class:`ModelCheckpointManager` for saving
        epoch and best-model checkpoints.
    :param scaler: :class:`~torch.cuda.amp.GradScaler` for AMP, or ``None``.
    :param use_amp: Whether to use automatic mixed precision.
    """
    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        clear_cache(device)

        train_loss, train_mlm, train_edge = train_epoch(
            model, train_dl, optimizer, scheduler, device, tracker, scaler, use_amp
        )

        clear_cache(device)

        val_loss, val_mlm, val_edge = validate(model, val_dl, device, tracker, use_amp)

        current_lr = optimizer.param_groups[0]["lr"]

        tracker.log_epoch(epoch, "train", train_loss, train_mlm, train_edge, current_lr)
        tracker.log_epoch(epoch, "val", val_loss, val_mlm, val_edge)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n  Peak GPU Memory: {peak_mem:.2f} GB")

        print_epoch_results(
            epoch,
            args,
            train_loss,
            train_mlm,
            train_edge,
            val_loss,
            val_mlm,
            val_edge,
            current_lr,
            tracker,
        )

        checkpoint_manager.save_checkpoint(model, args.tokenizer, epoch)

        if tracker.update_best(val_loss, epoch):
            print(f"\nNew best model! Saving best model...")
            checkpoint_manager.save_best_model(model, args.tokenizer)
        else:
            print(
                f"\nNo improvement. Patience: {tracker.patience_counter}/{args.early_stopping_patience}"
            )

        if tracker.should_stop_early():
            print(f"\nEarly stopping triggered!")
            print(f"   No improvement for {args.early_stopping_patience} epochs")
            print(
                f"   Best loss: {tracker.best_val_loss:.6f} "
                f"at epoch {tracker.history['best_epoch'] + 1}"
            )
            break

    print(f"\n{'=' * 70}")
    print(f"Training completed!")
    print(
        f"Best val loss: {tracker.best_val_loss:.6f} "
        f"at epoch {tracker.history['best_epoch'] + 1}"
    )
    print(f"{'=' * 70}\n")


def main() -> None:
    """Parse CLI arguments, configure training, and launch the training loop.

    Loads defaults from ``config.json`` via :func:`load_config_and_set_defaults`
    before parsing; CLI arguments override config values.  Validates that
    ``data_file`` and ``output_dir`` are provided (either via config or CLI).

    Full sequence:

    1. Build and configure the :class:`~argparse.ArgumentParser`.
    2. Apply config-file defaults and parse arguments.
    3. Validate required arguments.
    4. Set the global random seed via :func:`set_seed`.
    5. Select compute device via :func:`setup_device`.
    6. Instantiate :class:`PerformanceTracker` and
       :class:`ModelCheckpointManager`.
    7. Load model, tokenizer, and data loaders via :func:`setup_model_and_data`.
    8. Create optimiser and scheduler via
       :func:`setup_optimizer_and_scheduler`.
    9. Run the training loop via :func:`training_loop`.
    10. Persist all metrics via :meth:`PerformanceTracker.save`.
    """
    parser = argparse.ArgumentParser(
        description="Train GraphCodeBERT with Edge Prediction"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (relative to project root)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to load (e.g., Erlang trained model)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="Max sequence length"
    )
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "--mlm_probability", type=float, default=None, help="MLM masking probability"
    )
    parser.add_argument(
        "--validation_split", type=float, default=None, help="Validation split ratio"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_amp",
        default=True,
        action="store_true",
        help="Use mixed precision training",
    )

    project_root = load_config_and_set_defaults(parser)
    args = parser.parse_args()

    if not args.data_file:
        parser.error(
            "data_file must be specified in config.json or via --data_file argument."
        )
    if not args.output_dir:
        parser.error(
            "output_dir must be specified in config.json or via --output_dir argument."
        )

    set_seed(42)

    device, use_amp = setup_device()
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = PerformanceTracker(
        str(output_path), patience=args.early_stopping_patience
    )
    checkpoint_manager = ModelCheckpointManager(str(output_path))

    model, tokenizer, train_dl, val_dl = setup_model_and_data(
        args, device, project_root
    )
    args.tokenizer = tokenizer

    optimizer, scheduler = setup_optimizer_and_scheduler(model, args, train_dl)
    scaler = GradScaler() if use_amp else None

    print_training_config(args, device, use_amp)

    training_loop(
        model,
        train_dl,
        val_dl,
        optimizer,
        scheduler,
        device,
        args,
        tracker,
        checkpoint_manager,
        scaler,
        use_amp,
    )

    print("=" * 70)
    print("SAVING PERFORMANCE METRICS...")
    print("=" * 70)
    tracker.save()
    print(f"\nAll results saved to {output_path}")
    print(f"Best model saved at: {checkpoint_manager.get_best_model_path()}")
    print(f"Checkpoints saved at: {checkpoint_manager.checkpoints_dir}")


if __name__ == "__main__":
    main()
