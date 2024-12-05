# scripts/export_to_onnx.py

import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, setup_logging
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Export Model to ONNX")
    parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save the ONNX model', required=True)
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
    logger = setup_logging('logs/export_to_onnx.json.log')
    logger.info("Starting model export to ONNX...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)
    
    model.eval()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, config['model']['input_size']).to(device)
    complexities = {
        'variance': torch.tensor([0.6]),
        'entropy': torch.tensor([0.6]),
        'sparsity': torch.tensor([0.4])
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input, complexities),
        args.output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input', 'complexities'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'complexities': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX format at {args.output_path}")
    print(f"Model exported to ONNX format at {args.output_path}")

if __name__ == "__main__":
    main()
