import argparse
import torch
import yaml
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds


def parse_args():
    """
    Parses command-line arguments for training the model.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Dynamic Neural Network")
    parser.add_argument("--config", type=str, help="Path to training config file", required=True)
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


def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    """
    Trains the model using adaptive thresholds and dynamic architecture.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): The computing device (CPU/GPU).
        epochs (int): Number of training epochs.
    """
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, device)


def validate_model(model, val_loader, device):
    """
    Validates the model and logs accuracy.

    Args:
        model (torch.nn.Module): The trained neural network model.
        val_loader (DataLoader): Validation dataset loader.
        device (str): The computing device (CPU/GPU).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")


def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hybrid_thresholds = HybridThresholds()

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)

    # Generate random training dataset
    train_dataset = TensorDataset(
        torch.randn(500, config["model"]["input_size"]),
        torch.randint(0, 10, (500,))
    )
    val_dataset = TensorDataset(
        torch.randn(100, config["model"]["input_size"]),
        torch.randint(0, 10, (100,))
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Train model
    train_model(model, train_loader, val_loader, optimizer, device, config["training"]["epochs"])


if __name__ == "__main__":
    main()
