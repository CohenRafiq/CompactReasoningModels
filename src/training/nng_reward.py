import math
import torch
from src.training.base_trainer import BaseTrainer
from typing import Any, Dict, Tuple

class NNGRewardTrainer(BaseTrainer):
    
    def _train_step(self, batch: Any) -> torch.Tensor:
        inputs = batch[0].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, _, _, _ = self.criterion(outputs, inputs)
        loss.backward()
        self.optimizer.step()
        return loss

    def _test_step(self, batch: Any) -> Tuple[Dict[str, float], int]:
        inputs = batch[0].to(self.device)
        prediction = self.model(inputs)
        loss, row_loss, col_loss, clue_match = self.criterion(prediction, inputs)
        
        batch_size = inputs.size(0)
        return {
            "loss": loss.item(),
            "row_loss": row_loss.item(),
            "col_loss": col_loss.item(),
            "clue_match_sum": clue_match.sum().item(),
            "correct_sum": (clue_match == 1.0).sum().item()
        }, batch_size

    def _finalise_test_metrics(self, accumulated: Dict[str, float], n_batches: int, total_samples: int) -> Dict[str, float]:
        return {
            "test_accuracy": accumulated["correct_sum"] / total_samples if total_samples > 0 else 0.0,
            "test_loss": accumulated["loss"] / n_batches,
            "row_loss": accumulated["row_loss"] / n_batches,
            "col_loss": accumulated["col_loss"] / n_batches,
            "clue_match_percent": accumulated["clue_match_sum"] / total_samples if total_samples > 0 else 0.0,
        }

    def _log_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        if self.logger:
            self.logger.log_metrics({
                "train_loss": avg_train_loss,
                "test_loss": test_metrics["test_loss"],
                "test_accuracy": test_metrics["test_accuracy"],
                "clue_match_percent": test_metrics["clue_match_percent"],
                "row_loss": test_metrics["row_loss"],
                "col_loss": test_metrics["col_loss"],
                "epoch": epoch,
            })

    def _print_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        print(
            f"Epoch {epoch:02d} | Train loss: {avg_train_loss:.4f} "
            f"| Test loss: {test_metrics['test_loss']:.4f} | Test accuracy: {test_metrics['test_accuracy'] * 100:.3%} "
            f"| Clue‑match: {test_metrics['clue_match_percent']:.2f}%"
        )