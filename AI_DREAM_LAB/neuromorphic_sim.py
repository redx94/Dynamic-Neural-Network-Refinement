import torch
import torch.nn as nn
from typing import Dict, Tuple


class NeuromorphicCore(nn.Module):
    """
    Neuromorphic core model inspired by synaptic activity dynamics.
    """

    def __init__(self, input_dim: int, spike_threshold: float = 0.5):
        """
        Initializes the NeuromorphicCore.

        Args:
            input_dim (int): Input feature dimension.
            spike_threshold (float): Threshold for synaptic spike generation.
        """
        super().__init__()
        self.membrane_potential = nn.Parameter(torch.zeros(input_dim))
        self.spike_threshold = spike_threshold
        self.synaptic_weights = nn.Parameter(
            torch.rand(input_dim, input_dim) * 0.01
        )
        self.refractory_period = 5
        self.time_steps = 100

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Processes the input through a neuromorphic computation framework.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, Dict]: Output spikes and additional metrics.
        """
        spikes = []
        potentials = []

        for _ in range(self.time_steps):
            self.membrane_potential += torch.matmul(x, self.synaptic_weights)

            # Generate spikes
            spike_mask = self.membrane_potential >= self.spike_threshold
            spikes.append(spike_mask.float())

            # Reset spiked neurons
            self.membrane_potential[spike_mask] = 0

            potentials.append(self.membrane_potential.clone())

        return torch.stack(spikes), {
            'membrane_potentials': torch.stack(potentials),
            'synaptic_activity': self.synaptic_weights.abs().mean()
        }
