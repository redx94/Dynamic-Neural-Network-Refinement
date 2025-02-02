import argparse
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds


def parse_args():
    """
    Parses command-line arguments for quantizing the model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Quantize Dynamic Neural Network")
    parser.add_argument("--config", type=str, help="Path to training config file", required=True)
    parser.add_argument("--model_path", type=str, help="Path to the trained model", required=True)
    parser.add_argument("--quantized_model_path", type=str, help="Path to save the quantized model", required=True)
    return parser.parse_args()


def main():
    """
    Main function to perform model quantization.
    """
    args = parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid_thresholds = HybridThresholds()

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    torch.save(quantized_model, args.quantized_model_path)
    print(f"Quantized model saved at {args.quantized_model_path}")


if __name__ == "__main__":
    main()
