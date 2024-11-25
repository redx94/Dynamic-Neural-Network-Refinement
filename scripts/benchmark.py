# scripts/benchmark.py

import argparse
import torch
import time
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, setup_logging
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Model Performance")
    parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def benchmark_training(config, device):
    analyzer = Analyzer()
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = getattr(torch.nn, config['training']['loss_function'])()
    
    dummy_input = torch.randn(config['training']['batch_size'], config['model']['input_size']).to(device)
    dummy_labels = torch.randint(0, config['model']['output_size'], (config['training']['batch_size'],)).to(device)
    
    start_time = time.time()
    for _ in range(100):  # Number of iterations to benchmark
        complexities = analyzer.analyze(dummy_input)
        complexities = hybrid_thresholds(complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1)
        outputs = model(dummy_input, complexities)
        loss = loss_fn(outputs, dummy_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Training Benchmark: 100 iterations took {total_time:.2f} seconds.")

def benchmark_inference(config, model, device):
    analyzer = Analyzer()
    dummy_input = torch.randn(config['evaluation']['batch_size'], config['model']['input_size']).to(device)
    
    start_time = time.time()
    for _ in range(100):  # Number of iterations to benchmark
        complexities = analyzer.analyze(dummy_input)
        complexities = HybridThresholds(
            initial_thresholds=config['thresholds'],
            annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
            total_epochs=config['thresholds']['total_epochs']
        )(complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1)
        outputs = model(dummy_input, complexities)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Inference Benchmark: 100 iterations took {total_time:.2f} seconds.")

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)
    
    # Benchmark training
    benchmark_training(config, device)
    
    # Benchmark inference
    benchmark_inference(config, model, device)

if __name__ == "__main__":
    main()
