"""
Tests for Mixture of Experts module.
"""

import pytest
import torch
import torch.nn as nn
from src.mixture_of_experts import (
    Expert,
    SparseGate,
    MixtureOfExperts,
    MoEDynamicNetwork
)


class TestExpert:
    """Tests for Expert module."""

    def test_expert_initialization(self):
        """Test expert is initialized correctly."""
        expert = Expert(input_dim=128, hidden_dim=256, output_dim=64)
        assert isinstance(expert.network, nn.Sequential)

    def test_expert_forward(self):
        """Test expert forward pass."""
        expert = Expert(input_dim=128, hidden_dim=256, output_dim=64)
        x = torch.randn(32, 128)
        output = expert(x)
        assert output.shape == (32, 64)

    def test_expert_with_dropout(self):
        """Test expert with dropout."""
        expert = Expert(input_dim=128, hidden_dim=256, output_dim=64, dropout=0.3)
        expert.train()
        x = torch.randn(32, 128)

        # Multiple forward passes should give different results due to dropout
        outputs = [expert(x) for _ in range(3)]
        assert not torch.allclose(outputs[0], outputs[1])


class TestSparseGate:
    """Tests for SparseGate module."""

    def test_sparse_gate_initialization(self):
        """Test sparse gate initialization."""
        gate = SparseGate(input_dim=128, num_experts=8, top_k=2)
        assert gate.num_experts == 8
        assert gate.top_k == 2

    def test_sparse_gate_forward(self):
        """Test sparse gate forward pass."""
        gate = SparseGate(input_dim=128, num_experts=8, top_k=2)
        x = torch.randn(16, 128)

        gates, indices, load_balance_loss = gate(x, training=True)

        assert gates.shape == (16, 8)
        assert indices.shape == (16, 2)
        assert isinstance(load_balance_loss, torch.Tensor)
        assert load_balance_loss >= 0

    def test_sparse_gate_sparsity(self):
        """Test that sparse gate produces sparse output."""
        gate = SparseGate(input_dim=128, num_experts=8, top_k=2)
        x = torch.randn(16, 128)

        gates, indices, _ = gate(x, training=False)

        # Count non-zero elements per row
        nonzero_per_row = (gates > 0).sum(dim=1)
        assert (nonzero_per_row <= 2).all()  # At most top_k experts per sample


class TestMixtureOfExperts:
    """Tests for MixtureOfExperts module."""

    def test_moe_initialization(self):
        """Test MoE initialization."""
        moe = MixtureOfExperts(
            input_dim=128,
            output_dim=64,
            num_experts=4,
            expert_hidden_dim=256,
            top_k=2
        )
        assert len(moe.experts) == 4

    def test_moe_forward(self):
        """Test MoE forward pass."""
        moe = MixtureOfExperts(
            input_dim=128,
            output_dim=64,
            num_experts=4,
            expert_hidden_dim=256,
            top_k=2
        )
        x = torch.randn(16, 128)

        result = moe(x, training=True)

        assert 'output' in result
        assert 'gates' in result
        assert 'expert_indices' in result
        assert 'load_balance_loss' in result
        assert 'expert_utilization' in result

        assert result['output'].shape == (16, 64)

    def test_moe_hierarchical(self):
        """Test hierarchical MoE."""
        moe = MixtureOfExperts(
            input_dim=128,
            output_dim=64,
            num_experts=8,
            expert_hidden_dim=256,
            top_k=2,
            hierarchical=True,
            num_groups=2
        )
        x = torch.randn(16, 128)

        result = moe(x, training=True)

        assert result['output'].shape == (16, 64)
        assert result['gates'].shape == (16, 8)


class TestMoEDynamicNetwork:
    """Tests for MoEDynamicNetwork."""

    def test_moe_network_initialization(self):
        """Test MoE network initialization."""
        model = MoEDynamicNetwork(
            input_dim=784,
            num_classes=10,
            num_experts=4,
            expert_hidden_dim=128,
            top_k=2
        )
        assert isinstance(model, nn.Module)

    def test_moe_network_forward(self):
        """Test MoE network forward pass."""
        model = MoEDynamicNetwork(
            input_dim=784,
            num_classes=10,
            num_experts=4,
            expert_hidden_dim=128,
            top_k=2
        )
        x = torch.randn(8, 784)

        result = model(x, training=True)

        assert 'logits' in result
        assert 'moe_output' in result
        assert 'routing_stats' in result

        assert result['logits'].shape == (8, 10)

    def test_moe_network_with_complexities(self):
        """Test MoE network with complexity metrics."""
        model = MoEDynamicNetwork(
            input_dim=784,
            num_classes=10,
            num_experts=4
        )
        x = torch.randn(8, 784)
        complexities = {
            'variance': torch.tensor([0.6] * 8),
            'entropy': torch.tensor([0.7] * 8),
            'sparsity': torch.tensor([0.3] * 8)
        }

        result = model(x, complexities=complexities, training=True)

        assert result['logits'].shape == (8, 10)

    def test_routing_stats(self):
        """Test routing statistics collection."""
        model = MoEDynamicNetwork(
            input_dim=784,
            num_classes=10,
            num_experts=4
        )

        # Run several forward passes
        for _ in range(5):
            x = torch.randn(8, 784)
            model(x, training=True)

        stats = model.get_routing_stats()
        assert 'expert_usage' in stats
        assert len(stats['expert_usage']) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
