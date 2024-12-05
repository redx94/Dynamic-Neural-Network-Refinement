# scripts/generate_synthetic.py

import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, setup_logging
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Data")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_synthetic_data(model, input_dim, output_dir, categories=3):
    os.makedirs(output_dir, exist_ok=True)
    for complexity in ['simple', 'moderate', 'complex']:
        # Generate dummy input based on complexity
        if complexity == 'simple':
            input_tensor = torch.randn(100, input_dim) * 0.5
        elif complexity == 'moderate':
            input_tensor = torch.randn(100, input_dim)
        else:
            input_tensor = torch.randn(100, input_dim) * 1.5
        
        # Define dummy complexities
        complexities = {
            'variance': torch.tensor([0.6] * 100),
            'entropy': torch.tensor([0.6] * 100),
            'sparsity': torch.tensor([0.4] * 100)
        }
        
        # Generate outputs
        with torch.no_grad():
            outputs = model(input_tensor, complexities)
            preds = outputs.argmax(dim=1)
        
        # Save synthetic data
        synthetic_data_path = os.path.join(output_dir, f'synthetic_{complexity}.pth')
        torch.save({
            'input': input_tensor,
            'predictions': preds
        }, synthetic_data_path)
    
    return

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging('logs/generate_synthetic.json.log')
    logger.info("Starting synthetic data generation...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)
    
    # Generate synthetic data
    generate_synthetic_data(
        model=model,
        input_dim=config['model']['input_size'],
        output_dir=config['output']['synthetic_data_path']
    )
    
    logger.info(f"Synthetic data generated and saved to {config['output']['synthetic_data_path']}")

if __name__ == "__main__":
    main()
