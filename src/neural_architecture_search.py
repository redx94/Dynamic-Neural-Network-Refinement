import torch
import torch.nn as nn
from typing import Dict, Any
import random


class SearchableNetwork(nn.Module):
    """
    A neural network architecture that supports dynamic modifications for
    Neural Architecture Search (NAS).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.build_network(config)

    def build_network(self, config: Dict[str, Any]) -> None:
        """
        Builds the network dynamically based on the provided configuration.

        Args:
            config (Dict[str, Any]): Dictionary containing architecture settings.
        """
        input_dim = config['input_dim']
        for i in range(config['num_layers']):
            self.layers.append(nn.Linear(input_dim, config[f'layer_{i}_units']))
            self.layers.append(self.get_activation(config['activation']))
            input_dim = config[f'layer_{i}_units']

        self.layers.append(nn.Linear(input_dim, config['output_dim']))

    def get_activation(self, name: str) -> nn.Module:
        """
        Returns the activation function by name.

        Args:
            name (str): Name of the activation function.

        Returns:
            nn.Module: Activation function module.
        """
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralArchitectureSearch:
    """
    Implements a basic Neural Architecture Search (NAS) framework.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.best_model = None
        self.best_score = float('-inf')

    def search(self, train_loader, val_loader, num_trials=10):
        """
        Runs NAS with multiple architecture trials.

        Args:
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
            num_trials (int): Number of architecture configurations to evaluate.

        Returns:
            Dict[str, Any]: Best architecture configuration.
        """
        best_config = None

        for _ in range(num_trials):
            config = {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'num_layers': random.randint(2, 6),
                'activation': random.choice(['relu', 'leaky_relu', 'elu', 'tanh']),
                'learning_rate': random.uniform(1e-4, 1e-2)
            }

            for i in range(config['num_layers']):
                config[f'layer_{i}_units'] = random.randint(64, 512)

            model = SearchableNetwork(config)
            score = self.evaluate(model, val_loader)

            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                best_config = config

        return best_config

    def evaluate(self, model, val_loader):
        """
        Evaluates a model's performance on the validation set.

        Args:
            model (nn.Module): Model to evaluate.
            val_loader (DataLoader): Validation dataset.

        Returns:
            float: Validation accuracy.
        """
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total