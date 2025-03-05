import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.dynamic_nn import DynamicNN
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Learning rate scheduling
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0  # Track best validation accuracy
    best_val_loss = float('inf')

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
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}: "
                    f"Loss = {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        accuracy = validate_model(model, val_loader, device)
        scheduler.step()  # Update learning rate

        # Save the model if validation accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Model saved with improved validation accuracy.")

        # Adjust the model architecture based on validation loss
        if avg_loss >= best_val_loss:
            model.remove_layer()
            logger.info("Removed a layer due to increasing validation loss.")
        else:
            model.add_layer(128)
            logger.info("Added a layer due to decreasing validation loss.")
        best_val_loss = avg_loss


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
    return accuracy


if __name__ == "__main__":
    model = DynamicNN(784, [256, 128], 10)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, epochs=20)

    # Evaluate the model on a few sample images
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Get a few sample images from the validation set
    sample_images = []
    sample_labels = []
    for i in range(5):
        image, label = val_dataset[i]
        sample_images.append(image)
        sample_labels.append(label)

    # Make predictions on the sample images
    with torch.no_grad():
        sample_images = torch.stack(sample_images)
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)

    # Print the predictions
    print("\nSample Image Predictions:")
    for i in range(len(sample_images)):
        print(f"Image {i+1}: Predicted = {predicted[i]}, Actual = {sample_labels[i]}")
