"""
Tests for Reinforcement Learning Routing module.
"""

import pytest
import torch
from src.rl_routing import (
    RoutingPolicyNetwork,
    RoutingEnvironment,
    RLRoutingAgent,
    HybridRouter
)


class TestRoutingPolicyNetwork:
    """Tests for RoutingPolicyNetwork."""

    def test_policy_network_initialization(self):
        """Test policy network initialization."""
        policy = RoutingPolicyNetwork(
            complexity_dim=3,
            hidden_dim=128,
            num_routes=2
        )
        assert policy.num_routes == 2

    def test_policy_network_forward(self):
        """Test policy network forward pass."""
        policy = RoutingPolicyNetwork(complexity_dim=3, hidden_dim=128)
        complexities = torch.randn(8, 3)

        policy_dist, route_probs, state_value = policy(complexities)

        assert route_probs.shape == (8, 2)
        assert state_value.shape == (8, 1)
        assert torch.allclose(route_probs.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_policy_network_history(self):
        """Test policy network with state history."""
        policy = RoutingPolicyNetwork(complexity_dim=3, hidden_dim=128)
        policy.reset_history(batch_size=4)

        complexities = torch.randn(4, 3)
        _, _, _ = policy(complexities, use_history=True)

        assert policy.state_history is not None

    def test_policy_deterministic_vs_stochastic(self):
        """Test deterministic vs stochastic action selection."""
        policy = RoutingPolicyNetwork(complexity_dim=3, hidden_dim=128)
        complexities = torch.randn(4, 3)

        # Multiple passes should be consistent in deterministic mode
        policy.eval()
        with torch.no_grad():
            _, probs1, _ = policy(complexities)
            _, probs2, _ = policy(complexities)

        assert torch.allclose(probs1, probs2)


class TestRoutingEnvironment:
    """Tests for RoutingEnvironment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = RoutingEnvironment()
        assert env.cost_deep > env.cost_shallow

    def test_compute_reward_deep_path(self):
        """Test reward computation for deep path."""
        env = RoutingEnvironment()
        high_complexity = torch.tensor([0.8])

        reward, info = env.compute_reward(route=1, complexity=high_complexity)

        assert isinstance(reward, float)
        assert 'effective_accuracy' in info
        assert 'cost' in info
        assert info['cost'] == env.cost_deep

    def test_compute_reward_shallow_path(self):
        """Test reward computation for shallow path."""
        env = RoutingEnvironment()
        low_complexity = torch.tensor([0.2])

        reward, info = env.compute_reward(route=0, complexity=low_complexity)

        assert info['cost'] == env.cost_shallow

    def test_optimal_routing_bonus(self):
        """Test that optimal routing gives bonus."""
        env = RoutingEnvironment(complexity_threshold=0.5)

        # High complexity + deep path = optimal
        reward_optimal, _ = env.compute_reward(route=1, complexity=torch.tensor([0.7]))

        # High complexity + shallow path = suboptimal
        reward_suboptimal, _ = env.compute_reward(route=0, complexity=torch.tensor([0.7]))

        assert reward_optimal > reward_suboptimal


class TestRLRoutingAgent:
    """Tests for RLRoutingAgent."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = RLRoutingAgent()
        assert agent.policy is not None
        assert agent.optimizer is not None

    def test_select_route(self):
        """Test route selection."""
        agent = RLRoutingAgent()
        complexities = {
            'variance': torch.tensor(0.6),
            'entropy': torch.tensor(0.7),
            'sparsity': torch.tensor(0.3)
        }

        route, log_prob = agent.select_route(complexities, deterministic=True)

        assert route in [0, 1]
        assert isinstance(log_prob, torch.Tensor)

    def test_select_route_stochastic(self):
        """Test stochastic route selection."""
        agent = RLRoutingAgent()

        complexities = {
            'variance': torch.tensor(0.5),
            'entropy': torch.tensor(0.5),
            'sparsity': torch.tensor(0.5)
        }

        # Select routes multiple times
        routes = [agent.select_route(complexities, deterministic=False)[0] for _ in range(20)]

        # Should see some variation (though not guaranteed)
        # Just check that it runs without error
        assert all(r in [0, 1] for r in routes)

    def test_store_transition(self):
        """Test transition storage."""
        agent = RLRoutingAgent()

        state = torch.randn(3)
        agent.store_transition(
            state=state,
            action=1,
            reward=0.5,
            value=0.3,
            log_prob=torch.tensor(-0.5),
            done=False
        )

        assert len(agent.buffer['states']) == 1
        assert len(agent.buffer['actions']) == 1

    def test_compute_gae(self):
        """Test GAE computation."""
        agent = RLRoutingAgent()

        rewards = [0.1, 0.2, 0.3, 0.4]
        values = [0.2, 0.3, 0.35, 0.4]
        dones = [False, False, False, False]
        next_value = 0.5

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)

    def test_update(self):
        """Test policy update."""
        agent = RLRoutingAgent()

        # Store some transitions
        for i in range(10):
            state = torch.randn(3)
            complexities = {
                'variance': state[0],
                'entropy': state[1],
                'sparsity': state[2]
            }
            route, log_prob = agent.select_route(complexities)
            reward = 0.5 if route == 1 else 0.3

            agent.store_transition(
                state=state,
                action=route,
                reward=reward,
                value=0.3,
                log_prob=log_prob,
                done=False
            )

        metrics = agent.update(batch_size=4)

        # May be empty if not enough samples
        # Just check it runs without error
        assert isinstance(metrics, dict)

    def test_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent = RLRoutingAgent()

        save_path = str(tmp_path / "agent.pt")
        agent.save(save_path)

        new_agent = RLRoutingAgent()
        new_agent.load(save_path)

        # Check weights are the same
        for (name1, param1), (name2, param2) in zip(
            agent.policy.named_parameters(),
            new_agent.policy.named_parameters()
        ):
            assert torch.allclose(param1, param2)


class TestHybridRouter:
    """Tests for HybridRouter."""

    def test_hybrid_router_initialization(self):
        """Test hybrid router initialization."""
        router = HybridRouter()
        assert router.rl_agent is not None
        assert router.rl_weight == 0.5

    def test_hybrid_router_route(self):
        """Test hybrid routing decision."""
        router = HybridRouter(
            complexity_thresholds={'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        )

        complexities = {
            'variance': torch.tensor([0.7]),
            'entropy': torch.tensor([0.6]),
            'sparsity': torch.tensor([0.3])
        }

        route, info = router.route(complexities, deterministic=True)

        assert route in [0, 1]
        assert 'rule_based_route' in info
        assert 'rl_route' in info

    def test_hybrid_router_update_weight(self):
        """Test RL weight update."""
        router = HybridRouter(rl_weight=0.5)

        # Good performance should increase RL weight
        router.update_rl_weight(performance=0.95, baseline=0.9)
        assert router.rl_weight > 0.5

        # Bad performance should decrease RL weight
        router.update_rl_weight(performance=0.8, baseline=0.9)
        assert router.rl_weight < 0.55  # Should have decreased from previous

    def test_hybrid_router_extreme_complexity(self):
        """Test routing with extreme complexity values."""
        router = HybridRouter()

        # Very high complexity should favor deep path
        high_complexities = {
            'variance': torch.tensor([0.9]),
            'entropy': torch.tensor([0.9]),
            'sparsity': torch.tensor([0.1])
        }
        route_high, info_high = router.route(high_complexities, deterministic=True)

        # Very low complexity should favor shallow path
        low_complexities = {
            'variance': torch.tensor([0.1]),
            'entropy': torch.tensor([0.1]),
            'sparsity': torch.tensor([0.9])
        }
        route_low, info_low = router.route(low_complexities, deterministic=True)

        # Rule-based routes should differ
        assert info_high['rule_based_route'] != info_low['rule_based_route']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
