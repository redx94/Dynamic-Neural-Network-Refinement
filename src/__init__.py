"""
Dynamic Neural Network Refinement package.
This package contains modules for dynamic neural network architecture refinement.
"""

__version__ = "0.3.0"

from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.layers import BaseLayer
from src.hybrid_thresholds import HybridThresholds

# New advanced modules
from src.mixture_of_experts import MixtureOfExperts, MoEDynamicNetwork, Expert, SparseGate
from src.rl_routing import RLRoutingAgent, RoutingPolicyNetwork, HybridRouter, RoutingEnvironment
from src.neural_architecture_search import (
    NASController,
    DifferentiableNASCell,
    EvolutionarySearcher,
    PerformancePredictor,
    ArchitectureConfig,
    SearchSpace
)
from src.pretrained import (
    PretrainedLoader,
    PretrainedModelRegistry,
    FineTuner,
    BenchmarkDataset,
    BenchmarkRunner,
    ModelCheckpoint
)

__all__ = [
    # Core modules
    'DynamicNeuralNetwork',
    'Analyzer',
    'BaseLayer',
    'HybridThresholds',
    # Mixture of Experts
    'MixtureOfExperts',
    'MoEDynamicNetwork',
    'Expert',
    'SparseGate',
    # RL Routing
    'RLRoutingAgent',
    'RoutingPolicyNetwork',
    'HybridRouter',
    'RoutingEnvironment',
    # Neural Architecture Search
    'NASController',
    'DifferentiableNASCell',
    'EvolutionarySearcher',
    'PerformancePredictor',
    'ArchitectureConfig',
    'SearchSpace',
    # Pretrained models
    'PretrainedLoader',
    'PretrainedModelRegistry',
    'FineTuner',
    'BenchmarkDataset',
    'BenchmarkRunner',
    'ModelCheckpoint',
]
