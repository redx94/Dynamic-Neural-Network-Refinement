import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config/train_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(model, analyzer, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = config['training'].get('epochs', 10)
    lr = config['training'].get('learning_rate', 0.001)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    checkpoint_dir = config['output'].get('checkpoint_dir', 'models/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config['output'].get('final_model_path', 'models/final/model.pth')), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Flatten MNIST images from (B, 1, 28, 28) to (B, 784)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            
            # Analyze complexity metrics
            with torch.no_grad():
                complexities = analyzer.analyze(data)
            
            # Forward pass
            output = model(data, complexities)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % config['training'].get('log_interval', 100) == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")
        
        # Validation
        accuracy = validate_model(model, analyzer, val_loader, device)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = config['output'].get('final_model_path', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best validation accuracy: {accuracy:.2f}%. Model saved.")

def validate_model(model, analyzer, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            complexities = analyzer.analyze(data)
            output = model(data, complexities)
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100.0 * correct / total
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    try:
        config = load_config("config/train_config.yaml")
    except FileNotFoundError:
        # Fallback config if not found
        config = {
            'training': {'batch_size': 64, 'epochs': 5, 'learning_rate': 0.001, 'log_interval': 100},
            'output': {'checkpoint_dir': 'models/checkpoints', 'final_model_path': 'models/final/model.pth'}
        }
        
    model = DynamicNeuralNetwork(hybrid_thresholds=None)
    analyzer = Analyzer()

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    logger.info("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    logger.info("Starting training process...")
    # For quick testing, let's limit epochs if we run it during our fix to verify it works, otherwise 100
    config['training']['epochs'] = 2 # Short run to test
    train_model(model, analyzer, train_loader, val_loader, config)
    logger.info("Training complete.")
