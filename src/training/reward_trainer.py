import time
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from src.data.criterion.nonogram import soft_run_lengths


class RewardTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 20,
        logger: Optional[Any] = None,  # W&B or Null logger
        scheduler=None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.epochs = epochs
        self.logger = logger
        self.scheduler = scheduler
        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # If the logger supports model watching (e.g. wandb.watch)
        if hasattr(logger, "watch_model"):
            logger.watch_model(model)

    # ------------------------------------------------------------------
    # Metric: run‑length match (clue‑match)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_length_match(
        logits: torch.Tensor,
        inputs: torch.Tensor,
        tau: float = 1e-6,
    ) -> float:
        import math
        N = logits.shape[0]
        side = math.isqrt(logits.shape[1])

        # Binary grid: (N, H, W)
        binary = (torch.sigmoid(logits) > 0.5).float().reshape(N, side, side)

        # Recover clue tensors from flat inputs: (N, 2, H, max_runs)
        clues_reshaped = inputs.reshape(N, 2, side, -1)
        row_clues = clues_reshaped[:, 0]   # (N, H, max_runs)
        col_clues = clues_reshaped[:, 1]   # (N, W, max_runs)

        max_row_runs = row_clues.shape[2]
        max_col_runs = col_clues.shape[2]

        # Separate calls for rows and columns
        pred_row = soft_run_lengths(binary,              max_row_runs, tau=tau)  # (N, H, max_row_runs)
        pred_col = soft_run_lengths(binary.transpose(1, 2), max_col_runs, tau=tau)  # (N, W, max_col_runs)

        row_match = (pred_row.round() == row_clues).all(dim=2).float()
        col_match = (pred_col.round() == col_clues).all(dim=2).float()

        return (row_match.mean() + col_match.mean()) / 2.0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, log_every: int = 10):
        print("🚀 Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            start = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                # ---------------------------------------------------------
                # 1️⃣  Move everything to the correct device
                # ---------------------------------------------------------
                data_load_time = time.time() - start
                inputs = inputs.to(self.device)

                gpu_transfer_time = time.time() - start - data_load_time

                # ---------------------------------------------------------
                # 2️⃣  Forward / loss / backward
                # ---------------------------------------------------------
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # ---------------------------------------------------------
                # 3️⃣  Logging (batch‑wise)
                # ---------------------------------------------------------
                if batch_idx % log_every == 0 and self.logger:
                    self.logger.log_metrics(
                        {
                            "batch_loss": loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )

                # ---------------------------------------------------------
                # 4️⃣  Timing bookkeeping
                # ---------------------------------------------------------
                batch_time = time.time() - start
                start = time.time()          # reset for next iteration

            # -------------------------------------------------------------
            # 5️⃣  Scheduler step (if any)
            # -------------------------------------------------------------
            if self.scheduler:
                self.scheduler.step()

            # -------------------------------------------------------------
            # 6️⃣  Epoch‑level reporting
            # -------------------------------------------------------------
            print(
                f"Load: {data_load_time:.3f}s | Transfer: {gpu_transfer_time:.3f}s | Total: {batch_time:.3f}s"
            )
            avg_train_loss = epoch_loss / len(self.train_loader)

            # Updated test call – now returns (accuracy, loss, clue_match_percent)
            test_acc, test_loss, clue_match_pct = self.test()

            if self.logger:
                self.logger.log_metrics(
                    {
                        "train_loss": avg_train_loss,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "clue_match_percent": clue_match_pct,
                        "epoch": epoch,
                    }
                )

            print(
                f"Epoch {epoch:02d} | Train loss: {avg_train_loss:.4f} "
                f"| Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.3%} "
                f"| Clue‑match: {clue_match_pct:.2f}%"
            )

            # -------------------------------------------------------------
            # 7️⃣  Early stopping logic
            # -------------------------------------------------------------
            if self.patience is not None:
                if test_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = test_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"🛑 Early stopping triggered at epoch {epoch}")
                        return

    # ------------------------------------------------------------------
    # Evaluation loop – returns (test_accuracy, test_loss, clue_match_percent)
    # ------------------------------------------------------------------
    def test(self) -> Tuple[float, float, float]:
        """
        Evaluates the model on the test set.

        Returns
        -------
        test_accuracy : float
            Fraction of samples where **all** outputs (rounded to the nearest
            integer) exactly match the target inputs.
        test_loss : float
            Average loss over the test set.
        clue_match_percent : float
            Average percentage of row/column run‑length clues that match
            (i.e. the metric computed by `_run_length_match`).
        """
        self.model.eval()
        total_correct = 0  # for exact‑match accuracy
        total_loss = 0.0
        total_clue_match = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs.to(self.device)

                prediction = self.model(inputs)

                # ---- loss -------------------------------------------------
                loss = self.criterion(prediction, inputs)
                total_loss += loss.item()

                # ---- clue‑match (run‑length) -------------------------------
                clue_match = self._run_length_match(prediction, inputs)  # scalar in [0,1]
                total_clue_match += clue_match
                # correct only if clue match is 100% (i.e. all clues match)
                correct_per_sample = (clue_match == 1.0).float().sum()
                total_correct += correct_per_sample
                total_samples += inputs.size(0)

        # Guard against division by zero
        test_accuracy = total_correct / total_samples if total_samples else 0.0
        test_loss = total_loss / (total_samples / inputs.size(0)) if total_samples else 0.0
        clue_match_percent = (total_clue_match / (total_samples / inputs.size(0))) * 100 if total_samples else 0.0

        return test_accuracy, test_loss, clue_match_percent