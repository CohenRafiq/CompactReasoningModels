import torch
from typing import Any, Optional, Any

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 20,
        logger: Optional[Any] = None  # W&B or Null logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.logger = logger
        
        # Watch model if logger supports it
        if hasattr(logger, 'watch_model'):
            logger.watch_model(model)
    
    def train(self, log_every: int = 10):
        print("Starting training...")
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
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
            
            # Log epoch metrics - no step parameter
            avg_loss = epoch_loss / len(self.train_loader)
            test_acc, test_loss = self.test()
            
            if self.logger:
                self.logger.log_metrics({
                    "train_loss": avg_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "epoch": epoch
                })  # No step parameter!
            
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={test_acc:.4f}")
    
    def test(self) -> tuple[float, float]:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == targets).all(dim=1).sum().item()
                total += targets.size(0)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return correct / total if total > 0 else 0.0, total_loss / len(self.test_loader)