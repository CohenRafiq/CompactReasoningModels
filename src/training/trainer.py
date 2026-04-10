import torch

class Trainer:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()
            for batch in self.train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            test_loss = self.test()
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
        return avg_loss

    def test(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss
    
    def test_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        return accuracy