import torch
from src.training.base_trainer import BaseTrainer
from typing import Any, Dict, Tuple

class NNGSupervisedTrainer(BaseTrainer):
    
    def _train_step(self, batch: Any) -> torch.Tensor:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if outputs.shape != targets.shape:
            outputs = outputs.reshape(targets.shape[0], -1, *targets.shape[1:])
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _evaluation_step(self, batch: Any) -> Tuple[Dict[str, float], int]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)
        
        if outputs.shape == targets.shape:
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predictions == targets).all(dim=1).sum().item()
        else:
            outputs = outputs.reshape(targets.shape[0], -1, *targets.shape[1:])
            predictions = outputs.argmax(dim=1)
            correct = (predictions == targets).all(dim=1).sum().item()
            
        per_label_acc = (predictions == targets).float().mean().item()
        loss = self.criterion(outputs, targets)
        
        batch_size = targets.size(0)
        return {
            "correct": correct,
            "per_label_acc": per_label_acc,
            "loss": loss.item()
        }, batch_size

    def _finalise_metrics(self, accumulated: Dict[str, float], n_batches: int, total_samples: int) -> Dict[str, float]:
        return {
            "accuracy": accumulated["correct"] / total_samples if total_samples > 0 else 0.0,
            "loss": accumulated["loss"] / n_batches,
            "per_label_accuracy": accumulated["per_label_acc"] / n_batches
        }

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        if self.logger:
            self.logger.log_metrics({
                "train_loss": metrics["train_loss"],
                "train_accuracy": metrics["train_accuracy"],
                "test_loss": metrics["test_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "epoch": epoch,
                "per_label_accuracy": metrics.get("test_per_label_accuracy", metrics.get("per_label_accuracy", 0.0))
            })

    def _print_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        print(
            f"Epoch {epoch:02d}: "
            f"Train Loss={metrics['train_loss']:.4f}, Train Acc={metrics['train_accuracy']:.4f} | "
            f"Test Loss={metrics['test_loss']:.4f}, Test Acc={metrics['test_accuracy']:.4f}"
        )