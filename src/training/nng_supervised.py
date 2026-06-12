import torch
from src.training.base_trainer import BaseTrainer
from typing import Any, Dict, Tuple

class NNGSupervisedTrainer(BaseTrainer):
    
    def _train_step(self, batch: Any) -> torch.Tensor:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _test_step(self, batch: Any) -> Tuple[Dict[str, float], int]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predictions == targets).all(dim=1).sum().item()
        per_label_acc = (predictions == targets).float().mean().item()
        loss = self.criterion(outputs, targets)
        
        batch_size = targets.size(0)
        return {
            "correct": correct,
            "per_label_acc": per_label_acc,
            "loss": loss.item()
        }, batch_size

    def _finalise_test_metrics(self, accumulated: Dict[str, float], n_batches: int, total_samples: int) -> Dict[str, float]:
        return {
            "test_accuracy": accumulated["correct"] / total_samples if total_samples > 0 else 0.0,
            "test_loss": accumulated["loss"] / n_batches,
            "per_label_accuracy": accumulated["per_label_acc"] / n_batches
        }

    def _log_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        if self.logger:
            self.logger.log_metrics({
                "train_loss": avg_train_loss,
                "test_loss": test_metrics["test_loss"],
                "test_accuracy": test_metrics["test_accuracy"],
                "epoch": epoch,
                "per_label_accuracy": test_metrics["per_label_accuracy"]
            })

    def _print_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        print(f"Epoch {epoch:02d}: Loss={avg_train_loss:.4f}, Acc={test_metrics['test_accuracy']:.4f}")