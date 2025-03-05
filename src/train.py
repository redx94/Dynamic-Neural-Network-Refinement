import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.dynamic_nn import DynamicNN
from src.optimization.qpso import QPSOOptimizer
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

    # Initialize QPSO optimizer
    qpso_optimizer = QPSOOptimizer(model, n_particles=20, max_iter=10)
    criterion = nn.CrossEntropyLoss()
    scheduler = None # No learning rate scheduler for QPSO

    best_accuracy = 0.0  # Track best validation accuracy
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Forward pass
            output = model(data, complexities={}, train_loader=train_loader)
            loss = criterion(output, target)

            # QPSO optimization step
            qpso_optimizer.step(train_loader)

            # Get the best loss from the optimizer
            avg_loss = loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}: "
                    f"Loss = {loss.item():.4f}"
                )
        logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {loss.item():.4f}")

        accuracy = validate_model(model, val_loader, device)
        # scheduler.step()  # Update learning rate

        # Save the model if validation accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Model saved with improved validation accuracy.")

        # Adjust the model architecture based on validation loss
        if avg_loss >= best_val_loss:
            # Replace a layer with a new layer
            new_layer = nn.Linear(128, 64)  # Example: Replace with a linear layer
            layer_index = 1  # Example: Replace the second layer
            model._quantum_entangle_modules(model.layers[layer_index-1], new_layer)
            model.replace_layer(layer_index, new_layer)
            logger.info("Replaced a layer due to increasing validation loss.")
        else:
            model.add_layer(128)
            logger.info("Added a layer due to decreasing validation loss.")
        best_val_loss = avg_loss

        model._gemini_quantum_feedback()  # Call Gemini's quantum-assisted feedback loops
        model._quantum_optimize_performance()  # Call quantum performance optimization
        # model._gemini_quantum_predictive_processing()  # Call Gemini's quantum predictive processing
        model._ar_enhanced_security_audit()  # Call AR-enhanced security audit
        model._gemini_ar_penetration_testing()  # Call Gemini-enhanced AR penetration testing
        model._quantum_multi_platform_validation()  # Call quantum multi-platform validation
        model._ar_diagnostics()  # Call AR diagnostics


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
