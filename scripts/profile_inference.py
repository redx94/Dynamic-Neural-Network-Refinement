import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, setup_logging
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Inference Performance")
    parser.add_argument('--config', type=str, help='Path to evaluation config file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

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

    # Example inference step
    dummy_input = torch.randn(
        config['evaluation']['batch_size'], config['model']['input_size']
    ).to(device)
    complexities = analyzer.analyze(dummy_input)
    complexities = hybrid_thresholds(
        complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=1
    )

    # Profile inference
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("model_inference"):
            _ = model(dummy_input, complexities)  # Fixed unused variable issue

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("logs/inference_trace.json")

    print("Profiling completed. Trace saved to logs/inference_trace.json")


if __name__ == "__main__":
    main()