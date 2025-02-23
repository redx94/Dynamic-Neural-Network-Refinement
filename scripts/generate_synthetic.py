import argparse
import torch
import yaml
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds


def parse_args():
    """
    Parses command-line arguments for generating synthetic data.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Synthetic Data")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file",
        required=True
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the synthetic dataset",
        required=True
    )
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


def generate_synthetic_data(model, input_dim, output_dir, categories=3):
    """
    Generates synthetic data using the trained model.

    Args:
        model (torch.nn.Module): Trained neural network model.
        input_dim (int): Input dimensionality.
        output_dir (str): Directory to save synthetic data.
        categories (int): Number of output categories.
    """
    torch.makedirs(output_dir, exist_ok=True)

    input_tensor = torch.randn(100, input_dim) * 0.5
    complexities = {
        'variance': torch.rand(100) * 0.6,
        'entropy': torch.rand(100) * 0.6,
        'sparsity': torch.rand(100) * 0.4
    }

    # Generate outputs
    with torch.no_grad():
        outputs = model(input_tensor, complexities)
        predictions = outputs.argmax(dim=1)

    # Save synthetic data
    synthetic_data_path = f"{output_dir}/synthetic_data.pt"
    torch.save({'input': input_tensor, 'predictions': predictions}, synthetic_data_path)

    print(f"Synthetic data generated and saved to {output_dir}")


def main():
    """
    Main function to generate synthetic data using a trained model.
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

    # Generate synthetic data
    generate_synthetic_data(
        model=model,
        input_dim=config["model"]["input_size"],
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
