
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.neural_network import DynamicNeuralNetwork
from models.hybrid_thresholds import HybridThresholds
from models.analyzer import Analyzer
from training.trainer import Trainer

def load_config(config_path="config/train_config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_transform():
    """Create data transformation pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_data_loaders(config):
    """Initialize data loaders for training and validation."""
    transform = create_transform()
    
    train_dataset = datasets.MNIST(
        'data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        'data', 
        train=False, 
        transform=transform
    )
    
    loader_args = {
        'batch_size': config['training']['batch_size'],
        'num_workers': config['training']['data_loader']['num_workers']
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=config['training']['data_loader']['shuffle'],
        **loader_args
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )
    
    return train_loader, val_loader

def save_model(model, config):
    """Save the trained model."""
    model_path = Path(config['output']['final_model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    """Main training function."""
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    train_loader, val_loader = get_data_loaders(config)
    logger.info("Data loaders initialized")
    
    trainer = Trainer(config)
    logger.info("Trainer initialized")
    
    logger.info("Starting training process")
    history = trainer.train(train_loader, val_loader)
    
    save_model(trainer.model, config)
    return history

if __name__ == "__main__":
    main()
