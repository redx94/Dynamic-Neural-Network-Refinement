# scripts/evaluate.py

import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, setup_logging
import yaml
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dynamic Neural Network")
    parser.add_argument('--config', type=str, help='Path to evaluation configuration file', required=True)
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
    logger = setup_logging('logs/evaluate.json.log')
    logger.info("Starting evaluation process...")
    
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
    analyzer = Analyzer()
    
    # Load evaluation data
    data = pd.read_csv(config['evaluation']['data_path'])
    # Replace with actual data loading and preprocessing
    dummy_input = torch.tensor(data.values, dtype=torch.float32).to(device)
    dummy_labels = torch.randint(0, config['model']['output_size'], (dummy_input.size(0),)).to(device)
    
    with torch.no_grad():
        complexities = analyzer.analyze(dummy_input)
        complexities = hybrid_thresholds(complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1)
        outputs = model(dummy_input, complexities)
        loss = torch.nn.CrossEntropyLoss()(outputs, dummy_labels)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == dummy_labels).float().mean().item()
    
    # Save evaluation results
    evaluation_results = {
        'loss': loss.item(),
        'accuracy': accuracy
    }
    eval_results_path = config['evaluation']['output']['evaluation_results_path']
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
    pd.DataFrame([evaluation_results]).to_csv(eval_results_path, index=False)
    
    logger.info(f"Evaluation completed. Loss: {loss.item()}, Accuracy: {accuracy}")
    logger.info(f"Results saved to {eval_results_path}")

if __name__ == "__main__":
    main()
