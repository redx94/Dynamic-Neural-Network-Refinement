# scripts/profile_training.py

import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import setup_logging
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Profile Training Performance")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize logging
    logger = setup_logging('logs/profile_training.json.log')
    logger.info("Starting profiling of training process...")
    
    # Initialize components
    analyzer = Analyzer()
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).cuda() if torch.cuda.is_available() else DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = getattr(torch.nn, config['training']['loss_function'])()
    
    # Example training step
    dummy_input = torch.randn(config['training']['batch_size'], config['model']['input_size']).cuda() if torch.cuda.is_available() else torch.randn(config['training']['batch_size'], config['model']['input_size'])
    dummy_labels = torch.randint(0, config['model']['output_size'], (config['training']['batch_size'],)).cuda() if torch.cuda.is_available() else torch.randint(0, config['model']['output_size'], (config['training']['batch_size'],))
    
    complexities = analyzer.analyze(dummy_input)
    complexities = hybrid_thresholds(complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("model_inference"):
            outputs = model(dummy_input, complexities)
            loss = loss_fn(outputs, dummy_labels)
        
        with record_function("backward_pass"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("logs/training_trace.json")
    logger.info("Profiling completed. Trace saved to logs/training_trace.json")

if __name__ == "__main__":
    main()
