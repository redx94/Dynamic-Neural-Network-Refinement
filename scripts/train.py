import argparse
import torch
import yaml
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from src.per_sample_complexity import ComplexityAnalyzer
from loguru import logger

logger.add("training.log", rotation="500 MB", level="INFO")

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

def train_model(model, train_loader, val_loader, optimizer, device, epochs, complexity_analyzer):
    """
    Trains the model using adaptive thresholds and dynamic architecture.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): The computing device (CPU/GPU).
        epochs (int): Number of training epochs.
        complexity_analyzer (ComplexityAnalyzer): Complexity analyzer.
    """
    model.to(device)

    logger.info("Starting training epochs")
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = target.long()

            # Calculate complexities
            complexities = complexity_analyzer.calculate_complexities(data)

            optimizer.zero_grad()
            output = model(data, complexities)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, device)

        # Save model checkpoint
        checkpoint_path = f"{config['training']['checkpoint_dir']}/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        logger.info(f"Epoch {epoch+1}/{epochs} completed")

    # Save final model
    final_model_path = "models/final/model_v1.0_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

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
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")

@logger.catch
def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_thresholds = {
        'variance': 0.1,
        'entropy': 0.5,
        'sparsity': 0.8
    }
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=initial_thresholds,
        annealing_start_epoch=config["training"]["annealing_start_epoch"],
        total_epochs=config["training"]["epochs"]
    )

    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST normalization
        transforms.Lambda(lambda x: x.view(-1, 784)),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    logger.info("Starting training process")

    # Initialize ComplexityAnalyzer
    complexity_analyzer = ComplexityAnalyzer()

    # Train model
    train_model(model, train_loader, val_loader, optimizer, device, config["training"]["epochs"], complexity_analyzer)

    logger.info("Training process completed")

if __name__ == "__main__":
    main()
