import argparse
import torch
import time
import yaml
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Model Performance")
    parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def benchmark_training(config, device):
    analyzer = Analyzer()
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    start_time = time.time()
    for _ in range(100):  # Number of iterations to benchmark
        input_data = torch.randn(
            config['training']['batch_size'], config['model']['input_size']
        ).to(device)
        complexities = analyzer.analyze(input_data)
        complexities = hybrid_thresholds(
            complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1
        )
        outputs = model(input_data, complexities)  # Fixed unused variable issue
        loss = torch.nn.functional.mse_loss(
            outputs, torch.randn(config['training']['batch_size'], config['model']['output_size']).to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"Training Benchmark: 100 iterations took {total_time:.2f} seconds.")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DynamicNeuralNetwork().to(device)
    model = load_model(model, args.model_path, device)
    benchmark_training(config, device)


if __name__ == "__main__":
    main()