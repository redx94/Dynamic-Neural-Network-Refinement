# scripts/nas.py

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from src.nas import NAS
from scripts.utils import load_model, setup_logging
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural Architecture Search")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging('logs/nas.json.log')
    logger.info("Starting Neural Architecture Search (NAS)...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    base_model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    base_model = load_model(base_model, args.model_path, device)
    
    # Define search space (placeholder)
    search_space = {
        'add_layer': True,
        'remove_layer': True,
        'increase_units': True,
        'decrease_units': True
    }
    
    # Create dummy dataloader for NAS evaluation
    # Replace with actual validation data
    dummy_input = torch.randn(100, config['model']['input_size'])
    dummy_labels = torch.randint(0, config['model']['output_size'], (100,))
    dataset = TensorDataset(dummy_input, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Initialize NAS
    nas = NAS(base_model=base_model, search_space=search_space, device=device)
    
    # Run NAS
    best_model = nas.run(dataloader, generations=3, population_size=3)
    
    # Save the best model
    best_model_path = 'models/nas/best_model.pth'
    torch.save(best_model.state_dict(), best_model_path)
    logger.info(f"Best model saved at {best_model_path}")
    print(f"Best model saved at {best_model_path}")

if __name__ == "__main__":
    main()
