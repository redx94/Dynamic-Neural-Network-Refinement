"""
Reinforcement Learning-based routing for dynamic neural network paths.
Implements policy gradient methods for learning optimal routing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import math


class RoutingPolicyNetwork(nn.Module):
    """
    Neural network that learns optimal routing policies.
    Uses attention-based architecture for routing decisions.
    """

    def __init__(
        self,
        complexity_dim: int = 3,
        hidden_dim: int = 128,
        num_routes: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_routes = num_routes

        # Complexity feature encoder
        self.complexity_encoder = nn.Sequential(
            nn.Linear(complexity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Self-attention for complexity features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_routes)
        )

        # Value head for actor-critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # State history for temporal dependencies
        self.history_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.register_buffer('state_history', None)

    def forward(
        self,
        complexities: torch.Tensor,
        use_history: bool = True
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        """
        Compute routing policy distribution.

        Args:
            complexities: Tensor of shape (batch_size, complexity_dim)
            use_history: Whether to use temporal history

        Returns:
            policy_dist: Categorical distribution over routes
            route_probs: Route probabilities
            state_value: Estimated state value for critic
        """
        # Encode complexity features
        encoded = self.complexity_encoder(complexities)

        # Add sequence dimension for attention
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)

        # Self-attention over complexity features
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)

        # Incorporate temporal history
        if use_history and self.state_history is not None:
            attended_seq = attended.unsqueeze(1)
            attended_seq, self.state_history = self.history_rnn(attended_seq, self.state_history)
            attended = attended_seq.squeeze(1)

        # Compute policy and value
        logits = self.policy_head(attended)
        route_probs = F.softmax(logits, dim=-1)
        policy_dist = Categorical(route_probs)
        state_value = self.value_head(attended)

        return policy_dist, route_probs, state_value

    def reset_history(self, batch_size: int = 1):
        """Reset the state history buffer."""
        self.state_history = torch.zeros(1, batch_size, 128, device=next(self.parameters()).device)


class RoutingEnvironment:
    """
    RL environment for routing decision optimization.
    Simulates the tradeoff between computational cost and accuracy.
    """

    def __init__(
        self,
        cost_deep: float = 1.0,
        cost_shallow: float = 0.3,
        accuracy_deep: float = 0.95,
        accuracy_shallow: float = 0.85,
        complexity_threshold: float = 0.5
    ):
        self.cost_deep = cost_deep
        self.cost_shallow = cost_shallow
        self.accuracy_deep = accuracy_deep
        self.accuracy_shallow = accuracy_shallow
        self.complexity_threshold = complexity_threshold

    def compute_reward(
        self,
        route: int,
        complexity: torch.Tensor,
        actual_accuracy: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a routing decision.

        Args:
            route: 0 for shallow path, 1 for deep path
            complexity: Complexity metric (0-1 scale)
            actual_accuracy: Optional actual accuracy for supervised reward

        Returns:
            reward: Scalar reward
            info: Dictionary with reward components
        """
        is_deep = route == 1
        complexity_val = complexity.mean().item() if complexity.numel() > 1 else complexity.item()

        # Base accuracy for the chosen path
        base_accuracy = self.accuracy_deep if is_deep else self.accuracy_shallow

        # Complexity-adjusted accuracy
        # Deep path is better for high complexity, shallow for low complexity
        if is_deep:
            accuracy_modifier = complexity_val  # Better for high complexity
        else:
            accuracy_modifier = 1 - complexity_val  # Better for low complexity

        effective_accuracy = base_accuracy * (0.5 + 0.5 * accuracy_modifier)

        # Cost penalty
        cost = self.cost_deep if is_deep else self.cost_shallow

        # Reward = accuracy - cost_weight * normalized_cost
        cost_weight = 0.3
        reward = effective_accuracy - cost_weight * cost

        # Bonus for optimal routing
        optimal_route = 1 if complexity_val > self.complexity_threshold else 0
        if route == optimal_route:
            reward += 0.1  # Optimality bonus

        # Use actual accuracy if provided (for supervised fine-tuning)
        if actual_accuracy is not None:
            reward = actual_accuracy - cost_weight * cost

        return reward, {
            'effective_accuracy': effective_accuracy,
            'cost': cost,
            'optimal_route': optimal_route,
            'is_optimal': route == optimal_route
        }


class RLRoutingAgent:
    """
    RL agent that learns optimal routing decisions using policy gradients.
    Implements Proximal Policy Optimization (PPO) for stable learning.
    """

    def __init__(
        self,
        complexity_dim: int = 3,
        hidden_dim: int = 128,
        num_routes: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.policy = RoutingPolicyNetwork(
            complexity_dim=complexity_dim,
            hidden_dim=hidden_dim,
            num_routes=num_routes
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

        self.environment = RoutingEnvironment()

    def select_route(
        self,
        complexities: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Select routing action based on complexity metrics.

        Args:
            complexities: Dictionary with 'variance', 'entropy', 'sparsity'
            deterministic: If True, select most probable route

        Returns:
            route: Selected route (0 or 1)
            log_prob: Log probability of selected action
        """
        # Convert complexities to tensor
        complexity_tensor = torch.stack([
            complexities.get('variance', torch.tensor(0.5)),
            complexities.get('entropy', torch.tensor(0.5)),
            complexities.get('sparsity', torch.tensor(0.5))
        ]).view(1, -1)

        with torch.no_grad():
            policy_dist, route_probs, value = self.policy(complexity_tensor)

            if deterministic:
                route = route_probs.argmax(dim=-1).item()
                log_prob = torch.log(route_probs[0, route] + 1e-10)
            else:
                route = policy_dist.sample().item()
                log_prob = policy_dist.log_prob(torch.tensor(route))

        return route, log_prob

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
        done: bool
    ):
        """Store a transition in the experience buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.tensor(advantages), torch.tensor(returns)

    def update(self, batch_size: int = 64, num_epochs: int = 10) -> Dict[str, float]:
        """
        Update policy using PPO.

        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer['states']) < batch_size:
            return {}

        # Compute advantages
        states = torch.stack(self.buffer['states'])
        actions = torch.tensor(self.buffer['actions'])
        old_log_probs = torch.stack(self.buffer['log_probs'])

        with torch.no_grad():
            _, _, next_value = self.policy(states[-1:])
        advantages, returns = self.compute_gae(
            self.buffer['rewards'],
            self.buffer['values'],
            self.buffer['dones'],
            next_value.item()
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(num_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Get current policy outputs
                policy_dist, route_probs, values = self.policy(mb_states)
                new_log_probs = policy_dist.log_prob(mb_actions)
                entropy = policy_dist.entropy().mean()

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}

        return {
            'total_loss': total_loss / (num_epochs * (len(states) // batch_size + 1)),
            'policy_loss': sum(policy_losses) / len(policy_losses),
            'value_loss': sum(value_losses) / len(value_losses),
            'entropy': sum(entropies) / len(entropies)
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class HybridRouter:
    """
    Combines complexity-based routing with RL-learned policies.
    Provides smooth interpolation between rule-based and learned routing.
    """

    def __init__(
        self,
        rl_agent: Optional[RLRoutingAgent] = None,
        complexity_thresholds: Optional[Dict[str, float]] = None,
        rl_weight: float = 0.5
    ):
        self.rl_agent = rl_agent or RLRoutingAgent()
        self.complexity_thresholds = complexity_thresholds or {
            'variance': 0.5,
            'entropy': 0.5,
            'sparsity': 0.5
        }
        self.rl_weight = rl_weight

    def route(
        self,
        complexities: Dict[str, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, float]]:
        """
        Determine routing decision combining complexity rules and RL policy.

        Args:
            complexities: Dictionary of complexity metrics
            deterministic: Whether to use deterministic routing

        Returns:
            route: 0 for shallow, 1 for deep
            info: Routing information
        """
        # Rule-based routing
        variance = complexities.get('variance', torch.tensor(0.0)).mean().item()
        entropy = complexities.get('entropy', torch.tensor(0.0)).mean().item()
        sparsity = complexities.get('sparsity', torch.tensor(1.0)).mean().item()

        rule_based_route = 1 if (
            variance > self.complexity_thresholds['variance'] and
            entropy > self.complexity_thresholds['entropy'] and
            sparsity < self.complexity_thresholds['sparsity']
        ) else 0

        # RL-based routing
        rl_route, log_prob = self.rl_agent.select_route(complexities, deterministic)

        # Weighted combination
        if deterministic:
            # Use weighted voting
            route = 1 if (
                self.rl_weight * rl_route + (1 - self.rl_weight) * rule_based_route > 0.5
            ) else 0
        else:
            # Sample from combined distribution
            rl_prob = torch.exp(log_prob).item() if rl_route == 1 else 1 - torch.exp(log_prob).item()
            combined_prob = self.rl_weight * rl_prob + (1 - self.rl_weight) * rule_based_route
            route = 1 if torch.rand(1).item() < combined_prob else 0

        return route, {
            'rule_based_route': rule_based_route,
            'rl_route': rl_route,
            'combined_prob': self.rl_weight * rl_route + (1 - self.rl_weight) * rule_based_route,
            'complexity_scores': {
                'variance': variance,
                'entropy': entropy,
                'sparsity': sparsity
            }
        }

    def update_rl_weight(self, performance: float, baseline: float = 0.9):
        """
        Dynamically adjust RL weight based on performance.
        Increase RL weight when it outperforms baseline.
        """
        if performance > baseline:
            self.rl_weight = min(1.0, self.rl_weight + 0.05)
        else:
            self.rl_weight = max(0.0, self.rl_weight - 0.05)
