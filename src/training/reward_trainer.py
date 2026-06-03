import time
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from src.data.criterion.nonogram import grid_to_row_clues


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

    def _test_fixed_example(self):
        print("\n🔍 Testing on a fixed example to inspect outputs and clue matching...")
        example_clues = torch.tensor([[[[2,1,0], [2,1,0], [4,0,0], [0,0,0], [1,0,0]], [[1,0,0], [3,1,0], [1,1,0], [2,0,0], [1,1,0]]]], dtype=torch.float32).to(self.device)  # shape: (1, 2, 5, 3)
        example_clues = example_clues.flatten(start_dim=1)
        # Print Output grid, predicted clues, loss, and clue match percentage for a fixed example
        with torch.no_grad():
            output = self.model(example_clues)
            loss, row_loss, col_loss = self.criterion(output, example_clues)
            clue_match_pct = self._run_length_match(output, example_clues) * 100
            print(f"Output grid:\n{output.reshape(5,5)}")
            print(f"Predicted row clues:\n{grid_to_row_clues(output.reshape(1, 5, 5), K=3)}")
            print(f"Predicted column clues:\n{grid_to_row_clues(output.reshape(1, 5, 5).transpose(1, 2), K=3)}")
            print(loss)
            print(type(loss))
            print(loss.item())
            print(f"Loss: {loss.item():.4f} | Clue match percentage: {clue_match_pct:.2f}%")
            print(f"Row loss: {row_loss.item():.4f} | Column loss: {col_loss.item():.4f}")
            
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
        binary = (logits > 0.5).float().reshape(N, side, side)

        # Recover clue tensors from flat inputs: (N, 2, H, max_runs)
        clues_reshaped = inputs.reshape(N, 2, side, -1)
        row_clues = clues_reshaped[:, 0]   # (N, H, max_runs)
        col_clues = clues_reshaped[:, 1]   # (N, W, max_runs)

        max_row_runs = row_clues.shape[2]
        max_col_runs = col_clues.shape[2]

        # Separate calls for rows and columns
        pred_row = grid_to_row_clues(binary, max_row_runs)  # (N, H, max_row_runs)
        pred_col = grid_to_row_clues(binary.transpose(1, 2), max_col_runs)  # (N, W, max_col_runs)

        row_match = (pred_row.round() == row_clues).all(dim=2).float()
        col_match = (pred_col.round() == col_clues).all(dim=2).float()

        return (row_match.mean(dim=1) + col_match.mean(dim=1)) / 2.0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, log_every: int = 10):
        print("🚀 Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            start = time.time()
            total_data_load_time = 0.0
            total_gpu_transfer_time = 0.0
            total_batch_time = 0.0

            for batch_idx, inputs in enumerate(self.train_loader):
                # ---------------------------------------------------------
                # 1️⃣  Move everything to the correct device
                # ---------------------------------------------------------
                data_load_time = time.time() - start
                total_data_load_time += data_load_time
                inputs = inputs.to(self.device)

                gpu_transfer_time = time.time() - start - data_load_time
                total_gpu_transfer_time += gpu_transfer_time

                # ---------------------------------------------------------
                # 2️⃣  Forward / loss / backward
                # ---------------------------------------------------------
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, _, _ = self.criterion(outputs, inputs)
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
                total_batch_time += batch_time
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
                f"Load: {total_data_load_time:.3f}s | Transfer: {total_gpu_transfer_time:.3f}s | Total: {total_batch_time:.3f}s"
            )
            avg_train_loss = epoch_loss / len(self.train_loader)

            # Updated test call – now returns (accuracy, loss, clue_match_percent)
            test_acc, test_loss, clue_match_pct, row_loss, col_loss = self.test()

            if self.logger:
                self.logger.log_metrics(
                    {
                        "train_loss": avg_train_loss,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "clue_match_percent": clue_match_pct,
                        "row_loss": row_loss,
                        "col_loss": col_loss,
                        "epoch": epoch,
                    }
                )

            print(
                f"Epoch {epoch:02d} | Train loss: {avg_train_loss:.4f} "
                f"| Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.3%} "
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
                        self._test_fixed_example()
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
        total_row_loss = 0.0
        total_col_loss = 0.0

        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs.to(self.device)
                prediction = self.model(inputs)

                loss, row_loss, col_loss = self.criterion(prediction, inputs)
                total_loss += loss.item()
                total_row_loss += row_loss.item()
                total_col_loss += col_loss.item()

                clue_match = self._run_length_match(prediction, inputs)  # (N,)
                total_clue_match += clue_match.sum().item()
                total_correct += (clue_match == 1.0).sum().item()
                total_samples += inputs.size(0)

        n_batches = len(self.test_loader)
        test_accuracy = total_correct / total_samples
        test_loss = total_loss / n_batches
        row_loss = total_row_loss / n_batches
        col_loss = total_col_loss / n_batches
        clue_match_percent = total_clue_match / total_samples
        self.model.train()

        return test_accuracy, test_loss, clue_match_percent, row_loss, col_loss