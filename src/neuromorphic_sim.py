
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class NeuromorphicCore(nn.Module):
    def __init__(self, input_dim: int, spike_threshold: float = 0.5):
        super().__init__()
        self.membrane_potential = nn.Parameter(torch.zeros(input_dim))
        self.spike_threshold = spike_threshold
        self.synaptic_weights = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.refractory_period = 5
        self.time_steps = 100
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        spikes = []
        potentials = []
        
        for t in range(self.time_steps):
            self.membrane_potential += torch.matmul(x, self.synaptic_weights)
            spike_mask = self.membrane_potential >= self.spike_threshold
            spikes.append(spike_mask.float())
            self.membrane_potential[spike_mask] = 0
            potentials.append(self.membrane_potential.clone())
            
        return torch.stack(spikes), {
            'membrane_potentials': torch.stack(potentials),
            'synaptic_activity': self.synaptic_weights.abs().mean()
        }
