
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

class FederatedMetaLearner(nn.Module):
    def __init__(self, base_model: nn.Module, num_clients: int = 100):
        super().__init__()
        self.base_model = base_model
        self.num_clients = num_clients
        self.epsilon = 0.1  # Privacy budget
        self.local_models = [copy.deepcopy(base_model) for _ in range(num_clients)]
        self.sensitivity = 1.0
        
    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        noise_scale = self.sensitivity / self.epsilon
        return gradients + torch.normal(0, noise_scale, gradients.shape)
    
    def aggregate_models(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        for name, param in self.base_model.named_parameters():
            aggregated_grad = torch.stack([
                self.add_noise(update[name].grad) 
                for update in client_updates
            ]).mean(0)
            param.data -= 0.01 * aggregated_grad
