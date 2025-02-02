import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import logging
from models.neural_network import DynamicNeuralNetwork
from models.hybrid_thresholds import HybridThresholds
from models.analyzer import Analyzer


class Trainer:
    """
    Trainer class for the Dynamic Neural Network.
    """

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.model = DynamicNeuralNetwork(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['model']['hidden_dims'],
            output_dim=config['model']['output_dim']
        ).to(self.device)

        self.hybrid_thresholds = HybridThresholds(
            initial_thresholds=config['thresholds']['initial'],
            annealing_start_epoch=config['training']['annealing_start_epoch'],
            total_epochs=config['training']['epochs']
        )

        self.analyzer = Analyzer()

        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['training']['learning_rate']
        )

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): Training data loader.
            epoch (int): Current training epoch.

        Returns:
            Dict[str, float]: Training loss and accuracy.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Compute complexity metrics
            complexities = self.analyzer.analyze(data)
            thresholded = self.hybrid_thresholds(
                complexities['variance'],
                complexities['entropy'],
                complexities['sparsity'],
                epoch
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data, thresholded)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        self.logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validates the model.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            Dict[str, float]: Validation loss and accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Compute complexity metrics
                complexities = self.analyzer.analyze(data)
                thresholded = self.hybrid_thresholds(
                    complexities['variance'],
                    complexities['entropy'],
                    complexities['sparsity'],
                    current_epoch=0
                )

                outputs = self.model(data, thresholded)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        self.logger.info(f"Validation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, list]:
        """
        Runs the complete training process.

        Args:
            train_loader (DataLoader): Training dataset.
            val_loader (DataLoader): Validation dataset.

        Returns:
            Dict[str, list]: Training and validation performance history.
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.config['training']['epochs']):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            # Log training metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

        return history
