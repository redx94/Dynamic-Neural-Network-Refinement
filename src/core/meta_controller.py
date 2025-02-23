import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class MetaController(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_state = self.state_encoder(state)
        return self.action_head(encoded_state), self.value_head(encoded_state)
    
    def get_architecture_action(self, network_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state_vector = self._encode_network_state(network_state)
        action, value = self(state_vector)
        return {
            'layer_width_modifier': action[0].item(),
            'skip_connection_prob': action[1].item(),
            'learning_rate_adjust': action[2].item(),
            'value_estimate': value.item()
        }
        
    def _encode_network_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode current network architecture and performance metrics
        metrics = [
            state['loss'],
            state['gradient_norm'],
            state['layer_utilization'],
            state['complexity_score']
        ]
        return torch.tensor(metrics, dtype=torch.float32)
