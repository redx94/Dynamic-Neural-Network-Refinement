import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.dynamic_nn import DynamicNeuralNetwork
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, epochs=100):
    """
    Trains the model using adaptive thresholds, complexity analysis, and NAS.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        epochs (int): Number of training epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}: Loss = {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, device)


def validate_model(model, val_loader, device):
    """
    Validates the model and logs accuracy.

    Args:
        model (nn.Module): The trained neural network model.
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


if __name__ == "__main__":
    model = DynamicNeuralNetwork(input_dim=784, hidden_sizes=[256, 128], output_dim=10)

    # Load dataset
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(500, 784), torch.randint(0, 10, (500,))
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 784), torch.randint(0, 10, (100,))
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, epochs=20)
