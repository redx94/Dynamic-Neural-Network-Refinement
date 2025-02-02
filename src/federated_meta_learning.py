import torch
import torch.nn as nn
import copy
from typing import List, Dict


class FederatedMetaLearner(nn.Module):
    """
    Federated Meta Learning class for updating models across multiple clients.
    """

    def __init__(self, base_model: nn.Module, num_clients: int = 100):
        """
        Initializes the FederatedMetaLearner.

        Args:
            base_model (nn.Module): The base neural network model.
            num_clients (int): Number of clients participating in training.
        """
        super().__init__()
        self.base_model = base_model
        self.num_clients = num_clients
        self.epsilon = 0.1  # Privacy budget
        self.local_models = [copy.deepcopy(base_model) for _ in range(num_clients)]
        self.sensitivity = 1.0

    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Adds differential privacy noise to gradients.

        Args:
            gradients (torch.Tensor): Original gradients.

        Returns:
            torch.Tensor: Noisy gradients.
        """
        noise_scale = self.sensitivity / self.epsilon
        return gradients + torch.normal(0, noise_scale, gradients.shape)

    def aggregate_models(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        """
        Aggregates client models using Federated Averaging.

        Args:
            client_updates (List[Dict[str, torch.Tensor]]): List of model parameter updates.
        """
        for name, param in self.base_model.named_parameters():
            aggregated_grad = torch.stack([
                self.add_noise(update[name].grad) for update in client_updates
            ]).mean(0)

            param.data -= 0.01 * aggregated_grad  # Learning rate of 0.01
