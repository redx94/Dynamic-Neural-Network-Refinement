import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
import yaml


def parse_args():
    """
    Parses command-line arguments for profiling inference.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Profile Inference Performance")
    parser.add_argument("--config", type=str, help="Path to evaluation config file", required=True)
    parser.add_argument("--model_path", type=str, help="Path to the trained model", required=True)
    return parser.parse_args()


def load_config(config_path):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    """
    Main function to profile inference performance.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid_thresholds = HybridThresholds()

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    analyzer = Analyzer()

    # Example inference step
    dummy_input = torch.randn(
        config["evaluation"]["batch_size"], config["model"]["input_size"]
    ).to(device)

    complexities = analyzer.analyze(dummy_input)

    # Profile inference
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            _ = model(dummy_input, complexities)

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("logs/inference_trace.json")

    print("Profiling complete. Trace saved to logs/inference_trace.json")


if __name__ == "__main__":
    main()
