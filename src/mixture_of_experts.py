"""
Mixture of Experts (MoE) module for dynamic expert routing.
Implements sparse gating and load balancing for efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class Expert(nn.Module):
    """Individual expert network in the MoE architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SparseGate(nn.Module):
    """Sparse gating network for expert selection with load balancing."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std

        self.gate_network = nn.Linear(input_dim, num_experts)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sparse gating weights for expert selection.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            training: Whether in training mode (adds noise for exploration)

        Returns:
            gates: Sparse gating weights (batch_size, num_experts)
            indices: Selected expert indices (batch_size, top_k)
            load_balance_loss: Auxiliary loss for load balancing
        """
        logits = self.gate_network(x)

        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Create sparse gate weights
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Create full gate tensor
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(logits, gates)

        return gates, top_k_indices, load_balance_loss

    def _compute_load_balance_loss(self, logits: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss to encourage uniform expert utilization.
        Based on "Outrageously Large Neural Networks" (Shazeer et al., 2017).
        """
        # Importance: sum of gates per expert
        importance = gates.sum(dim=0)

        # Load: probability of routing to each expert
        load = F.softmax(logits, dim=-1).sum(dim=0)

        # Coefficient of variation loss
        cv_importance = importance.std() / (importance.mean() + 1e-6)
        cv_load = load.std() / (load.mean() + 1e-6)

        return cv_importance + cv_load


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with sparse gating and dynamic routing.
    Supports both standard MoE and hierarchical MoE architectures.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 256,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 0.1,
        hierarchical: bool = False,
        num_groups: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hierarchical = hierarchical
        self.num_groups = num_groups

        if hierarchical:
            # Create groups of experts
            experts_per_group = num_experts // num_groups
            self.expert_groups = nn.ModuleList([
                nn.ModuleList([
                    Expert(input_dim, expert_hidden_dim, output_dim, dropout)
                    for _ in range(experts_per_group)
                ]) for _ in range(num_groups)
            ])
            self.group_gate = SparseGate(input_dim, num_groups, min(top_k, num_groups), noise_std)
            self.expert_gates = nn.ModuleList([
                SparseGate(input_dim, experts_per_group, top_k, noise_std)
                for _ in range(num_groups)
            ])
        else:
            # Standard flat MoE
            self.experts = nn.ModuleList([
                Expert(input_dim, expert_hidden_dim, output_dim, dropout)
                for _ in range(num_experts)
            ])
            self.gate = SparseGate(input_dim, num_experts, top_k, noise_std)

        self.router_weights: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch_size, input_dim)
            training: Whether in training mode

        Returns:
            Dictionary containing output, routing info, and auxiliary losses
        """
        if self.hierarchical:
            return self._forward_hierarchical(x, training)
        return self._forward_standard(x, training)

    def _forward_standard(self, x: torch.Tensor, training: bool) -> Dict[str, torch.Tensor]:
        """Standard flat MoE forward pass."""
        batch_size = x.shape[0]

        gates, indices, load_balance_loss = self.gate(x, training)
        self.router_weights['gates'] = gates
        self.router_weights['indices'] = indices

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weight expert outputs by gates
        output = torch.bmm(gates.unsqueeze(1), expert_outputs).squeeze(1)

        return {
            'output': output,
            'gates': gates,
            'expert_indices': indices,
            'load_balance_loss': load_balance_loss,
            'expert_utilization': (gates > 0).float().mean(dim=0)
        }

    def _forward_hierarchical(self, x: torch.Tensor, training: bool) -> Dict[str, torch.Tensor]:
        """Hierarchical MoE forward pass with group-level routing."""
        batch_size = x.shape[0]

        # First level: select expert groups
        group_gates, group_indices, group_loss = self.group_gate(x, training)

        # Second level: select experts within selected groups
        outputs = torch.zeros(x.shape[0], self.experts[0].network[-1].out_features, device=x.device)
        total_load_balance_loss = group_loss
        all_gates = torch.zeros(batch_size, self.num_experts, device=x.device)

        expert_idx = 0
        for g, (expert_group, expert_gate) in enumerate(zip(self.expert_groups, self.expert_gates)):
            expert_gates_g, indices_g, loss_g = expert_gate(x, training)
            total_load_balance_loss = total_load_balance_loss + loss_g

            # Weight by both group and expert gates
            combined_gates = group_gates[:, g:g+1] * expert_gates_g

            for i, expert in enumerate(expert_group):
                all_gates[:, expert_idx] = combined_gates[:, i]
                expert_output = expert(x)
                outputs = outputs + combined_gates[:, i:i+1] * expert_output
                expert_idx += 1

        return {
            'output': outputs,
            'gates': all_gates,
            'group_indices': group_indices,
            'load_balance_loss': total_load_balance_loss,
            'expert_utilization': (all_gates > 0).float().mean(dim=0)
        }


class MoEDynamicNetwork(nn.Module):
    """
    Dynamic Neural Network with MoE-based routing.
    Combines complexity-based routing with expert specialization.
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        num_experts: int = 8,
        expert_hidden_dim: int = 256,
        top_k: int = 2,
        complexity_thresholds: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.complexity_thresholds = complexity_thresholds or {
            'variance': 0.5,
            'entropy': 0.5,
            'sparsity': 0.5
        }

        # Input projection
        self.input_proj = nn.Linear(input_dim, expert_hidden_dim)

        # MoE layers
        self.moe_layer = MixtureOfExperts(
            input_dim=expert_hidden_dim,
            output_dim=expert_hidden_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k
        )

        # Output head
        self.output_head = nn.Linear(expert_hidden_dim, num_classes)

        # Routing statistics
        self.register_buffer('routing_history', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        complexities: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic MoE routing.

        Args:
            x: Input tensor
            complexities: Optional complexity metrics for adaptive routing
            training: Whether in training mode

        Returns:
            Dictionary with output and routing information
        """
        # Project input
        h = F.gelu(self.input_proj(x))

        # Pass through MoE
        moe_output = self.moe_layer(h, training)

        # Update routing statistics
        if training:
            self.routing_history = self.routing_history + moe_output['expert_utilization'].detach()
            self.total_samples += 1

        # Final output
        output = self.output_head(moe_output['output'])

        return {
            'logits': output,
            'moe_output': moe_output,
            'routing_stats': self.get_routing_stats()
        }

    def get_routing_stats(self) -> Dict[str, float]:
        """Get normalized routing statistics."""
        if self.total_samples == 0:
            return {'expert_usage': torch.zeros(self.moe_layer.num_experts).tolist()}

        normalized_usage = self.routing_history / self.total_samples
        return {'expert_usage': normalized_usage.tolist()}

    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_history.zero_()
        self.total_samples.zero_()
