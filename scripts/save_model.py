import argparse
import torch
import yaml
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, save_model


def parse_args():
    """
    Parses command-line arguments for saving a trained model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Save PyTorch Model")
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--model_path", type=str, help="Path to the trained model", required=True)
    parser.add_argument("--save_path", type=str, help="Path to save the model", required=True)
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
    Main function to load and save the model.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds']['initial'],
        annealing_start_epoch=config['training']['annealing_start_epoch'],
        total_epochs=config['training']['epochs']
    )

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model = load_model(model, args.model_path, device)

    # Save model
    save_model(model, args.save_path)
    print(f"Model saved at {args.save_path}")


if __name__ == "__main__":
    main()
