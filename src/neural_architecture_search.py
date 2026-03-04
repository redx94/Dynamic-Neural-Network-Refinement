"""
Advanced Neural Architecture Search (NAS) module.
Implements differentiable NAS, evolutionary search, and performance prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import random
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ArchitectureConfig:
    """Configuration for a neural network architecture."""
    num_layers: int
    hidden_dims: List[int]
    layer_types: List[str]
    activation: str
    dropout_rate: float
    skip_connections: List[Tuple[int, int]]

    def to_vector(self) -> torch.Tensor:
        """Convert architecture to a fixed-size vector representation."""
        max_layers = 10
        max_hidden = 512

        features = []

        # Normalize layer count
        features.append(self.num_layers / max_layers)

        # Hidden dimensions (padded)
        hidden_normalized = [d / max_hidden for d in self.hidden_dims[:max_layers]]
        hidden_normalized += [0] * (max_layers - len(hidden_normalized))
        features.extend(hidden_normalized)

        # Layer type encoding (one-hot per layer)
        layer_type_map = {'BaseLayer': 0, 'Linear': 1, 'Attention': 2, 'MoE': 3}
        for lt in self.layer_types[:max_layers]:
            encoding = [0] * 4
            if lt in layer_type_map:
                encoding[layer_type_map[lt]] = 1
            features.extend(encoding)

        # Activation encoding
        activation_map = {'relu': 0, 'gelu': 1, 'tanh': 2, 'silu': 3}
        act_encoding = [0] * 4
        if self.activation in activation_map:
            act_encoding[activation_map[self.activation]] = 1
        features.extend(act_encoding)

        return torch.tensor(features, dtype=torch.float32)


class SearchSpace:
    """Defines the search space for NAS operations."""

    def __init__(self):
        self.layer_types = ['BaseLayer', 'Linear', 'Attention', 'MoE']
        self.activations = ['relu', 'gelu', 'tanh', 'silu']
        self.hidden_dim_range = (32, 512)
        self.max_layers = 10
        self.dropout_range = (0.0, 0.5)

    def sample_random(self) -> ArchitectureConfig:
        """Sample a random architecture from the search space."""
        num_layers = random.randint(2, self.max_layers)

        hidden_dims = []
        layer_types = []
        current_dim = random.randint(*self.hidden_dim_range)

        for _ in range(num_layers):
            hidden_dims.append(current_dim)
            layer_types.append(random.choice(self.layer_types))
            current_dim = max(32, current_dim + random.randint(-64, 64))

        # Generate skip connections
        skip_connections = []
        if num_layers > 3:
            for i in range(num_layers - 2):
                if random.random() < 0.3:  # 30% chance of skip connection
                    target = random.randint(i + 2, num_layers - 1)
                    skip_connections.append((i, target))

        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            layer_types=layer_types,
            activation=random.choice(self.activations),
            dropout_rate=random.uniform(*self.dropout_range),
            skip_connections=skip_connections
        )


class DifferentiableNASCell(nn.Module):
    """
    Differentiable architecture cell for DARTS-style search.
    Implements continuous relaxation of architecture choices.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_operations: int = 4,
        num_nodes: int = 4
    ):
        super().__init__()
        self.num_operations = num_operations
        self.num_nodes = num_nodes

        # Operations
        self.operations = nn.ModuleList([
            nn.Linear(input_dim, output_dim),
            nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU()),
            nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU()),
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
        ])

        # Architecture parameters (alpha)
        self.alpha = nn.Parameter(torch.randn(num_nodes, num_nodes, num_operations) * 0.01)

        # Node transformations
        self.node_transforms = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else output_dim, output_dim)
            for i in range(num_nodes)
        ])

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with architecture weights.

        Args:
            x: Input tensor
            temperature: Temperature for softmax (lower = more discrete)

        Returns:
            Output tensor
        """
        # Initialize node states
        nodes = [self.node_transforms[0](x)]

        for i in range(1, self.num_nodes):
            # Aggregate inputs from previous nodes
            node_input = torch.zeros_like(nodes[0])
            for j in range(i):
                # Softmax over operations for edge (j -> i)
                edge_weights = F.softmax(self.alpha[j, i, :] / temperature, dim=-1)

                # Weighted sum of operations
                for k, op in enumerate(self.operations):
                    node_input = node_input + edge_weights[k] * op(nodes[j])

            nodes.append(self.node_transforms[i](node_input))

        return nodes[-1]

    def get_architecture(self) -> List[Tuple[int, int, int]]:
        """Extract discrete architecture from learned weights."""
        architecture = []
        for i in range(1, self.num_nodes):
            for j in range(i):
                best_op = self.alpha[j, i, :].argmax().item()
                if self.alpha[j, i, best_op] > 0:  # Only include significant edges
                    architecture.append((j, i, best_op))
        return architecture


class PerformancePredictor(nn.Module):
    """
    Predicts model performance from architecture configuration.
    Uses a transformer-based architecture for prediction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(46, embed_dim),  # Fixed size from ArchitectureConfig.to_vector()
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads
        self.accuracy_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.latency_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.memory_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, arch_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict performance metrics for an architecture.

        Args:
            arch_vector: Architecture representation tensor

        Returns:
            Dictionary of predicted metrics
        """
        # Embed input
        embedded = self.input_embed(arch_vector)

        # Add sequence dimension if needed
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1)

        # Transform
        transformed = self.transformer(embedded)
        transformed = transformed.mean(dim=1)  # Global pooling

        # Predict metrics
        return {
            'predicted_accuracy': self.accuracy_head(transformed).squeeze(-1),
            'predicted_latency_ms': self.latency_head(transformed).squeeze(-1),
            'predicted_memory_mb': self.memory_head(transformed).squeeze(-1)
        }


class EvolutionarySearcher:
    """
    Evolutionary algorithm for neural architecture search.
    Implements tournament selection, crossover, and mutation.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        elitism: int = 5
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.population: List[Tuple[ArchitectureConfig, float]] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

    def initialize_population(self):
        """Create initial random population."""
        self.population = [
            (self.search_space.sample_random(), 0.0)
            for _ in range(self.population_size)
        ]
        self.generation = 0

    def evaluate_population(
        self,
        evaluate_fn: Callable[[ArchitectureConfig], float]
    ):
        """Evaluate all architectures in the population."""
        evaluated_population = []
        for arch, _ in self.population:
            fitness = evaluate_fn(arch)
            evaluated_population.append((arch, fitness))
        self.population = evaluated_population

        # Record statistics
        fitnesses = [f for _, f in self.population]
        self.history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness': math.sqrt(sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))
        })

    def tournament_selection(self) -> ArchitectureConfig:
        """Select architecture using tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def crossover(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig
    ) -> ArchitectureConfig:
        """Perform crossover between two parent architectures."""
        if random.random() > self.crossover_rate:
            return parent1 if random.random() < 0.5 else parent2

        # Crossover layer count
        num_layers = (parent1.num_layers + parent2.num_layers) // 2

        # Crossover hidden dimensions
        hidden_dims = []
        for i in range(num_layers):
            if i < len(parent1.hidden_dims) and i < len(parent2.hidden_dims):
                hidden_dims.append(
                    parent1.hidden_dims[i] if random.random() < 0.5 else parent2.hidden_dims[i]
                )
            elif i < len(parent1.hidden_dims):
                hidden_dims.append(parent1.hidden_dims[i])
            elif i < len(parent2.hidden_dims):
                hidden_dims.append(parent2.hidden_dims[i])
            else:
                hidden_dims.append(random.randint(32, 512))

        # Crossover layer types
        layer_types = []
        for i in range(num_layers):
            if i < len(parent1.layer_types) and i < len(parent2.layer_types):
                layer_types.append(
                    parent1.layer_types[i] if random.random() < 0.5 else parent2.layer_types[i]
                )
            else:
                layer_types.append(random.choice(self.search_space.layer_types))

        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            layer_types=layer_types,
            activation=parent1.activation if random.random() < 0.5 else parent2.activation,
            dropout_rate=(parent1.dropout_rate + parent2.dropout_rate) / 2,
            skip_connections=parent1.skip_connections if random.random() < 0.5 else parent2.skip_connections
        )

    def mutate(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Apply mutation to an architecture."""
        if random.random() > self.mutation_rate:
            return architecture

        mutated = ArchitectureConfig(
            num_layers=architecture.num_layers,
            hidden_dims=architecture.hidden_dims.copy(),
            layer_types=architecture.layer_types.copy(),
            activation=architecture.activation,
            dropout_rate=architecture.dropout_rate,
            skip_connections=architecture.skip_connections.copy()
        )

        # Mutate number of layers
        if random.random() < 0.2:
            change = random.choice([-1, 1])
            mutated.num_layers = max(2, min(self.search_space.max_layers, mutated.num_layers + change))
            # Adjust hidden_dims and layer_types
            if change > 0:
                mutated.hidden_dims.append(random.randint(*self.search_space.hidden_dim_range))
                mutated.layer_types.append(random.choice(self.search_space.layer_types))
            else:
                mutated.hidden_dims.pop()
                mutated.layer_types.pop()

        # Mutate hidden dimensions
        if random.random() < 0.3:
            idx = random.randint(0, len(mutated.hidden_dims) - 1)
            mutated.hidden_dims[idx] = random.randint(*self.search_space.hidden_dim_range)

        # Mutate layer types
        if random.random() < 0.2:
            idx = random.randint(0, len(mutated.layer_types) - 1)
            mutated.layer_types[idx] = random.choice(self.search_space.layer_types)

        # Mutate activation
        if random.random() < 0.1:
            mutated.activation = random.choice(self.search_space.activations)

        # Mutate dropout
        if random.random() < 0.2:
            mutated.dropout_rate = random.uniform(*self.search_space.dropout_range)

        return mutated

    def evolve(self) -> ArchitectureConfig:
        """Perform one generation of evolution."""
        # Sort by fitness
        self.population.sort(key=lambda x: x[1], reverse=True)

        new_population = []

        # Elitism: keep top performers
        for i in range(min(self.elitism, len(self.population))):
            new_population.append((self.population[i][0], 0.0))

        # Create rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            new_population.append((child, 0.0))

        self.population = new_population
        self.generation += 1

        return self.population[0][0]  # Return best architecture

    def get_best_architecture(self) -> Tuple[ArchitectureConfig, float]:
        """Get the best architecture from the population."""
        if not self.population:
            raise ValueError("Population is empty. Initialize and evaluate first.")
        return max(self.population, key=lambda x: x[1])


class NASController:
    """
    Main controller for Neural Architecture Search.
    Coordinates different search strategies and performance prediction.
    """

    def __init__(
        self,
        search_method: str = 'evolutionary',
        use_predictor: bool = True,
        predictor_path: Optional[str] = None
    ):
        self.search_space = SearchSpace()
        self.search_method = search_method
        self.use_predictor = use_predictor

        # Initialize searcher
        if search_method == 'evolutionary':
            self.searcher = EvolutionarySearcher(self.search_space)
        elif search_method == 'differentiable':
            self.diff_cell = DifferentiableNASCell(784, 256)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

        # Initialize performance predictor
        if use_predictor:
            self.predictor = PerformancePredictor()
            if predictor_path:
                self.predictor.load_state_dict(torch.load(predictor_path))
        else:
            self.predictor = None

        self.search_history: List[Dict[str, Any]] = []

    def search(
        self,
        evaluate_fn: Callable[[ArchitectureConfig], float],
        num_iterations: int = 100,
        predictor_warmup: int = 10
    ) -> ArchitectureConfig:
        """
        Run architecture search.

        Args:
            evaluate_fn: Function to evaluate architecture performance
            num_iterations: Number of search iterations
            predictor_warmup: Iterations before using predictor

        Returns:
            Best found architecture
        """
        if self.search_method == 'evolutionary':
            return self._evolutionary_search(evaluate_fn, num_iterations, predictor_warmup)
        else:
            return self._differentiable_search(evaluate_fn, num_iterations)

    def _evolutionary_search(
        self,
        evaluate_fn: Callable[[ArchitectureConfig], float],
        num_generations: int,
        predictor_warmup: int
    ) -> ArchitectureConfig:
        """Run evolutionary architecture search."""
        self.searcher.initialize_population()

        for gen in range(num_generations):
            # Evaluate population
            def predict_or_evaluate(arch: ArchitectureConfig) -> float:
                if self.predictor and gen >= predictor_warmup:
                    # Use predictor with some actual evaluations for diversity
                    if random.random() < 0.2:
                        actual_fitness = evaluate_fn(arch)
                        # Update predictor training data
                        return actual_fitness
                    else:
                        with torch.no_grad():
                            pred = self.predictor(arch.to_vector().unsqueeze(0))
                            return pred['predicted_accuracy'].item()
                else:
                    return evaluate_fn(arch)

            self.searcher.evaluate_population(predict_or_evaluate)
            best_arch = self.searcher.evolve()

            # Record search progress
            self.search_history.append({
                'generation': gen,
                'best_fitness': self.searcher.history[-1]['best_fitness'],
                'mean_fitness': self.searcher.history[-1]['mean_fitness']
            })

        return self.searcher.get_best_architecture()[0]

    def _differentiable_search(
        self,
        evaluate_fn: Callable[[ArchitectureConfig], float],
        num_epochs: int
    ) -> ArchitectureConfig:
        """Run differentiable architecture search (DARTS-style)."""
        optimizer = torch.optim.Adam(self.diff_cell.parameters(), lr=0.01)
        arch_optimizer = torch.optim.Adam([self.diff_cell.alpha], lr=0.001)

        for epoch in range(num_epochs):
            # Training step (would need actual data loader)
            # This is a simplified version

            # Temperature annealing
            temperature = max(0.1, 1.0 - epoch / num_epochs)

            # Architecture optimization
            arch_optimizer.zero_grad()
            # In practice, compute validation loss here
            arch_loss = -self.diff_cell.alpha.mean()  # Placeholder
            arch_loss.backward()
            arch_optimizer.step()

            # Weight optimization
            optimizer.zero_grad()
            # In practice, compute training loss here
            # train_loss.backward()
            optimizer.step()

        # Extract final architecture
        arch = self.diff_cell.get_architecture()
        return self._architecture_from_darts(arch)

    def _architecture_from_darts(self, darts_arch: List[Tuple[int, int, int]]) -> ArchitectureConfig:
        """Convert DARTS architecture to ArchitectureConfig."""
        num_nodes = max(max(edge[0], edge[1]) for edge in darts_arch) + 1 if darts_arch else 4

        return ArchitectureConfig(
            num_layers=num_nodes,
            hidden_dims=[256] * num_nodes,
            layer_types=['BaseLayer'] * num_nodes,
            activation='relu',
            dropout_rate=0.1,
            skip_connections=[(e[0], e[1]) for e in darts_arch]
        )

    def save_predictor(self, path: str):
        """Save the performance predictor model."""
        if self.predictor:
            torch.save(self.predictor.state_dict(), path)

    def load_predictor(self, path: str):
        """Load a pre-trained performance predictor."""
        if self.predictor:
            self.predictor.load_state_dict(torch.load(path))
