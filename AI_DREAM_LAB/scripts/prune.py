# scripts/prune.py

import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, save_model, setup_logging
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Prune Dynamic Neural Network")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--pruned_model_path', type=str, help='Path to save the pruned model', default='models/pruned/pruned_model.pth')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prune_model(model, pruning_percentage=0.2):
    parameters_to_prune = (
        (model.layer1.layer, 'weight'),
        (model.layer2.layer, 'weight'),
        (model.layer3.layer, 'weight'),
    )
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=pruning_percentage,
    )
    return model

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging('logs/prune.json.log')
    logger.info("Starting pruning process...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)
    
    # Prune the model
    pruned_model = prune_model(model, pruning_percentage=0.2)
    
    # Save pruned model
    os.makedirs(os.path.dirname(args.pruned_model_path), exist_ok=True)
    save_model(pruned_model, args.pruned_model_path)
    logger.info(f"Pruned model saved at {args.pruned_model_path}")

if __name__ == "__main__":
    main()
