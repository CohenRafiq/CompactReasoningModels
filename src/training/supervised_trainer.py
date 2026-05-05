import torch
from typing import Any, Optional, Any
import time

class SupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 20,
        logger: Optional[Any] = None,  # W&B or Null logger
        scheduler=None, 
        early_stopping_patience=None, 
        early_stopping_min_delta=1e-4
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.logger = logger
        self.scheduler = scheduler
        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0


        # Watch model if logger supports it
        if hasattr(logger, 'watch_model'):
            logger.watch_model(model)
    
    def train(self, log_every: int = 10):
        print("Starting training...")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                data_load_time = time.time() - start
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                gpu_transfer_time = time.time() - start - data_load_time
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log without step - W&B auto-increments
                if batch_idx % log_every == 0 and self.logger:
                    self.logger.log_metrics({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })  # No step parameter!
                batch_time = time.time() - start
                start = time.time()
            
            if self.scheduler:
                self.scheduler.step()

            print(f"Load: {data_load_time:.3f}s | Transfer: {gpu_transfer_time:.3f}s | Total: {batch_time:.3f}s")
            # Log epoch metrics - no step parameter
            avg_loss = epoch_loss / len(self.train_loader)
            test_acc, test_loss, per_label_acc = self.test()

            if self.logger:
                self.logger.log_metrics({
                    "train_loss": avg_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "epoch": epoch,
                    "per_label_accuracy": per_label_acc
                })  # No step parameter!
            
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={test_acc:.4f}")
            if self.patience:
                if test_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = test_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        return
        
    def test(self) -> tuple[float, float, float]:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        total_per_label = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == targets).all(dim=1).sum().item()
                total_per_label += (predictions == targets).float().mean().item()
                total += targets.size(0)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        per_label_acc = total_per_label / len(self.test_loader)
        return correct / total if total > 0 else 0.0, total_loss / len(self.test_loader), per_label_acc