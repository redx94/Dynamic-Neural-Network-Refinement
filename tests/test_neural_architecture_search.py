"""
Tests for Neural Architecture Search module.
"""

import pytest
import torch
from src.neural_architecture_search import (
    ArchitectureConfig,
    SearchSpace,
    DifferentiableNASCell,
    PerformancePredictor,
    EvolutionarySearcher,
    NASController
)


class TestArchitectureConfig:
    """Tests for ArchitectureConfig."""

    def test_config_creation(self):
        """Test creating architecture configuration."""
        config = ArchitectureConfig(
            num_layers=4,
            hidden_dims=[256, 256, 128, 64],
            layer_types=['BaseLayer'] * 4,
            activation='relu',
            dropout_rate=0.1,
            skip_connections=[(0, 2)]
        )

        assert config.num_layers == 4
        assert len(config.hidden_dims) == 4

    def test_config_to_vector(self):
        """Test converting config to vector."""
        config = ArchitectureConfig(
            num_layers=4,
            hidden_dims=[256, 256, 128, 64],
            layer_types=['BaseLayer', 'Linear', 'BaseLayer', 'Linear'],
            activation='gelu',
            dropout_rate=0.1,
            skip_connections=[]
        )

        vector = config.to_vector()

        assert isinstance(vector, torch.Tensor)
        assert vector.dim() == 1


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_search_space_initialization(self):
        """Test search space initialization."""
        space = SearchSpace()
        assert len(space.layer_types) > 0
        assert len(space.activations) > 0

    def test_sample_random(self):
        """Test random architecture sampling."""
        space = SearchSpace()

        config = space.sample_random()

        assert isinstance(config, ArchitectureConfig)
        assert 2 <= config.num_layers <= space.max_layers
        assert len(config.hidden_dims) == config.num_layers

    def test_sample_multiple(self):
        """Test sampling multiple architectures."""
        space = SearchSpace()

        configs = [space.sample_random() for _ in range(10)]

        # Should see some variety
        num_layers_set = {c.num_layers for c in configs}
        assert len(num_layers_set) > 1  # At least some variety


class TestDifferentiableNASCell:
    """Tests for DifferentiableNASCell."""

    def test_cell_initialization(self):
        """Test NAS cell initialization."""
        cell = DifferentiableNASCell(
            input_dim=128,
            output_dim=64,
            num_operations=4,
            num_nodes=4
        )

        assert cell.num_operations == 4
        assert cell.num_nodes == 4

    def test_cell_forward(self):
        """Test NAS cell forward pass."""
        cell = DifferentiableNASCell(input_dim=128, output_dim=64)
        x = torch.randn(8, 128)

        output = cell(x, temperature=1.0)

        assert output.shape == (8, 64)

    def test_cell_temperature_effect(self):
        """Test effect of temperature on cell output."""
        cell = DifferentiableNASCell(input_dim=128, output_dim=64)
        x = torch.randn(8, 128)

        # High temperature = more continuous
        output_high_temp = cell(x, temperature=2.0)

        # Low temperature = more discrete
        output_low_temp = cell(x, temperature=0.1)

        # Both should produce valid outputs
        assert output_high_temp.shape == (8, 64)
        assert output_low_temp.shape == (8, 64)

    def test_get_architecture(self):
        """Test extracting architecture from cell."""
        cell = DifferentiableNASCell(input_dim=128, output_dim=64)

        arch = cell.get_architecture()

        assert isinstance(arch, list)
        # Architecture should be list of (from_node, to_node, operation) tuples
        for edge in arch:
            assert len(edge) == 3


class TestPerformancePredictor:
    """Tests for PerformancePredictor."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = PerformancePredictor()
        assert predictor is not None

    def test_predictor_forward(self):
        """Test predictor forward pass."""
        predictor = PerformancePredictor()
        arch_vector = torch.randn(46)  # Fixed size from ArchitectureConfig

        predictions = predictor(arch_vector.unsqueeze(0))

        assert 'predicted_accuracy' in predictions
        assert 'predicted_latency_ms' in predictions
        assert 'predicted_memory_mb' in predictions

        # Accuracy should be in [0, 1]
        assert 0 <= predictions['predicted_accuracy'].item() <= 1

    def test_predictor_batch(self):
        """Test predictor with batch input."""
        predictor = PerformancePredictor()
        arch_vectors = torch.randn(16, 46)

        predictions = predictor(arch_vectors)

        assert predictions['predicted_accuracy'].shape == (16,)


class TestEvolutionarySearcher:
    """Tests for EvolutionarySearcher."""

    def test_searcher_initialization(self):
        """Test evolutionary searcher initialization."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space)

        assert searcher.population_size == 50
        assert searcher.mutation_rate == 0.1

    def test_initialize_population(self):
        """Test population initialization."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, population_size=20)

        searcher.initialize_population()

        assert len(searcher.population) == 20

    def test_evaluate_population(self):
        """Test population evaluation."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, population_size=10)
        searcher.initialize_population()

        # Simple evaluation function
        def evaluate(arch):
            return sum(arch.hidden_dims) / 1000  # Dummy fitness

        searcher.evaluate_population(evaluate)

        # All architectures should have fitness values
        for arch, fitness in searcher.population:
            assert isinstance(fitness, float)

    def test_tournament_selection(self):
        """Test tournament selection."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, population_size=20)
        searcher.initialize_population()

        # Add dummy fitness
        searcher.population = [(arch, i * 0.1) for i, (arch, _) in enumerate(searcher.population)]

        selected = searcher.tournament_selection()

        assert isinstance(selected, ArchitectureConfig)

    def test_crossover(self):
        """Test crossover operation."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, crossover_rate=1.0)  # Always crossover

        parent1 = space.sample_random()
        parent2 = space.sample_random()

        child = searcher.crossover(parent1, parent2)

        assert isinstance(child, ArchitectureConfig)
        assert child.num_layers <= max(parent1.num_layers, parent2.num_layers) + 1

    def test_mutate(self):
        """Test mutation operation."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, mutation_rate=1.0)  # Always mutate

        original = space.sample_random()
        mutated = searcher.mutate(original)

        assert isinstance(mutated, ArchitectureConfig)
        # Some aspect should be different (highly likely with mutation_rate=1.0)

    def test_evolve(self):
        """Test one generation of evolution."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, population_size=10)
        searcher.initialize_population()

        def evaluate(arch):
            return sum(arch.hidden_dims) / 1000

        searcher.evaluate_population(evaluate)
        best_arch = searcher.evolve()

        assert isinstance(best_arch, ArchitectureConfig)
        assert searcher.generation == 1

    def test_get_best_architecture(self):
        """Test getting best architecture."""
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, population_size=10)
        searcher.initialize_population()

        def evaluate(arch):
            return arch.num_layers / 10  # More layers = better fitness

        searcher.evaluate_population(evaluate)

        best_arch, best_fitness = searcher.get_best_architecture()

        assert isinstance(best_arch, ArchitectureConfig)
        assert isinstance(best_fitness, float)


class TestNASController:
    """Tests for NASController."""

    def test_controller_initialization_evolutionary(self):
        """Test controller initialization with evolutionary search."""
        controller = NASController(search_method='evolutionary')
        assert isinstance(controller.searcher, EvolutionarySearcher)

    def test_controller_initialization_differentiable(self):
        """Test controller initialization with differentiable search."""
        controller = NASController(search_method='differentiable')
        assert controller.diff_cell is not None

    def test_controller_invalid_method(self):
        """Test controller with invalid search method."""
        with pytest.raises(ValueError):
            NASController(search_method='invalid')

    def test_controller_search_evolutionary(self):
        """Test evolutionary search through controller."""
        controller = NASController(
            search_method='evolutionary',
            use_predictor=False
        )

        # Simple evaluation function
        call_count = [0]

        def evaluate(arch):
            call_count[0] += 1
            return sum(arch.hidden_dims) / 1000

        best_arch = controller.search(evaluate, num_iterations=5)

        assert isinstance(best_arch, ArchitectureConfig)
        assert call_count[0] > 0

    def test_controller_with_predictor(self):
        """Test controller with performance predictor."""
        controller = NASController(
            search_method='evolutionary',
            use_predictor=True
        )

        def evaluate(arch):
            return sum(arch.hidden_dims) / 1000

        best_arch = controller.search(evaluate, num_iterations=10, predictor_warmup=5)

        assert isinstance(best_arch, ArchitectureConfig)

    def test_save_load_predictor(self, tmp_path):
        """Test saving and loading predictor."""
        controller = NASController(use_predictor=True)

        save_path = str(tmp_path / "predictor.pt")
        controller.save_predictor(save_path)

        new_controller = NASController(use_predictor=True)
        new_controller.load_predictor(save_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
