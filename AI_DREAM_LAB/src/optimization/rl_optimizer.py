import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from collections import deque

class ArchitectureOptimizer:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 memory_size: int = 10000):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.memory = deque(maxlen=memory_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(state)
            noise = torch.randn_like(action) * 0.1
            return torch.clamp(action + noise, -1, 1)
            
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.memory) < batch_size:
            return {}
            
        transitions = self._sample_memory(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = transitions
        
        # Update critic
        current_q = self.critic(torch.cat([state_batch, action_batch], dim=1))
        next_actions = self.actor(next_state_batch)
        next_q = self.critic(torch.cat([next_state_batch, next_actions], dim=1))
        target_q = reward_batch + 0.99 * next_q
        
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(torch.cat([
            state_batch,
            self.actor(state_batch)
        ], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
