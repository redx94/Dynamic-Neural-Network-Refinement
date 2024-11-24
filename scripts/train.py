import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_model(model, train_loader, optimizer, total_epochs):
    for epoch in range(total_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
        print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
