import argparse
import torch
import yaml
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds


def parse_args():
    """
    Parses command-line arguments for exporting the model to ONNX.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Export Model to ONNX")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        required=True
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model",
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the ONNX model",
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


def main():
    """
    Main function to export a trained model to ONNX format.
    """
    args = parse_args()
    config = load_config(args.config)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid_thresholds = HybridThresholds()

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Prepare input for tracing
    dummy_input = torch.randn(1, config["model"]["input_size"]).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"Model exported to ONNX format at {args.output_path}")


if __name__ == "__main__":
    main()
