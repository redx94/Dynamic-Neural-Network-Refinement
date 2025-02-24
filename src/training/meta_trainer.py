import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from collections import deque

class MetaControllerTrainer:
    def __init__(self, 
                 meta_controller: nn.Module,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2):
        self.meta_controller = meta_controller
        self.optimizer = torch.optim.Adam(meta_controller.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.experience_buffer = deque(maxlen=10000)
        
    def collect_experience(self, 
                         state: Dict[str, torch.Tensor],
                         action: Dict[str, float],
                         reward: float,
                         next_state: Dict[str, torch.Tensor],
                         done: bool):
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update_policy(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.experience_buffer) < batch_size:
            return {}
            
        batch = self._sample_batch(batch_size)
        advantages = self._compute_advantages(batch)
        
        # PPO policy update
        for _ in range(3):  # Multiple epochs of policy optimization
            loss_metrics = self._optimize_step(batch, advantages)
            
        return loss_metrics
    
    def _compute_advantages(self, batch: Dict) -> torch.Tensor:
        with torch.no_grad():
            values = self.meta_controller.value_head(
                self.meta_controller.state_encoder(batch['states'])
            )
            next_values = self.meta_controller.value_head(
                self.meta_controller.state_encoder(batch['next_states'])
            )
            
            # GAE advantage computation
            advantages = torch.zeros_like(values)
            gae = 0
            for t in reversed(range(len(batch['rewards']))):
                delta = (batch['rewards'][t] + 
                        self.gamma * next_values[t] * (1 - batch['dones'][t]) - 
                        values[t])
                gae = delta + self.gamma * self.gae_lambda * (1 - batch['dones'][t]) * gae
                advantages[t] = gae
                
        return advantages
