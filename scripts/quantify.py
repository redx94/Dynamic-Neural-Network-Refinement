# scripts/quantify.py

import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, save_model, setup_logging
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Dynamic Neural Network")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--quantized_model_path', type=str, help='Path to save the quantized model', required=True)
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
    logger = setup_logging('logs/quantify.json.log')
    logger.info("Starting model quantization process...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)
    
    # Quantize the model (Dynamic Quantization)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Save quantized model
    save_model(quantized_model, args.quantized_model_path)
    logger.info(f"Quantized model saved at {args.quantized_model_path}")
    print(f"Quantized model saved at {args.quantized_model_path}")

if __name__ == "__main__":
    main()
