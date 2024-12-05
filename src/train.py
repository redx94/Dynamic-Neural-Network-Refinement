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

def load_config():
    """Load training configuration from YAML file."""
    config_path = Path("config/train_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_data_loaders(config):
    """Initialize data loaders for training and validation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Using MNIST as an example dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['data_loader']['shuffle'],
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Set up data loaders
    train_loader, val_loader = get_data_loaders(config)
    logger.info("Data loaders initialized")
    
    # Initialize trainer with configuration
    trainer = Trainer(config)
    logger.info("Trainer initialized")
    
    # Start training
    logger.info("Starting training process")
    history = trainer.train(train_loader, val_loader)
    
    # Save the trained model
    model_path = Path(config['output']['final_model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return history

if __name__ == "__main__":
    main()
