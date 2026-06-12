import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader


class BaseTrainer:
    """
    Base trainer handling the common training loop, early stopping, 
    and logging infrastructure. Subclasses must implement the abstract hooks.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int,
        logger: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-4,
        print_every: int = 1,
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
        self.print_every = print_every
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        if hasattr(logger, "watch_model"):
            logger.watch_model(model)

    def train(self, log_every: int = 10) -> None:
        if self.print_every > 0:
            print("Starting training...")
            
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                epoch_loss += loss.item()

                if batch_idx % log_every == 0 and self.logger:
                    self.logger.log_metrics({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    })

            if self.scheduler:
                self.scheduler.step()

            avg_train_loss = epoch_loss / len(self.train_loader)
            
            test_metrics = self.test()

            self._log_epoch(epoch, avg_train_loss, test_metrics)
            if self.print_every > 0 and epoch % self.print_every == 0:
                self._print_epoch(epoch, avg_train_loss, test_metrics)

            # Early stopping logic
            test_loss = test_metrics.get("test_loss", float("inf"))
            test_acc = test_metrics.get("test_accuracy", 0.0)

            if self.patience is not None:
                if test_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = test_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience or test_acc >= 1.0:
                        if self.print_every > 0:
                            print(f"🛑 Early stopping triggered at epoch {epoch}")
                        return

    def test(self) -> Dict[str, float]:
        """
        Executes the test loop. Subclasses define how to process a batch 
        and how to aggregate the results.
        """
        self.model.eval()
        accumulated_metrics: Dict[str, float] = {}
        total_samples = 0
        n_batches = len(self.test_loader)

        with torch.no_grad():
            for batch in self.test_loader:
                batch_metrics, batch_size = self._test_step(batch)
                total_samples += batch_size
                
                for key, value in batch_metrics.items():
                    accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value

        return self._finalise_test_metrics(accumulated_metrics, n_batches, total_samples)

    # --- Abstract Hooks to be implemented by subclasses ---
    
    def _train_step(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _train_step")

    def _test_step(self, batch: Any) -> Tuple[Dict[str, float], int]:
        raise NotImplementedError("Subclasses must implement _test_step")

    def _finalise_test_metrics(self, accumulated: Dict[str, float], n_batches: int, total_samples: int) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement _finalize_test_metrics")

    def _log_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        raise NotImplementedError("Subclasses must implement _log_epoch")

    def _print_epoch(self, epoch: int, avg_train_loss: float, test_metrics: Dict[str, float]) -> None:
        raise NotImplementedError("Subclasses must implement _print_epoch")