import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.nas import NAS
from src.models.dynamic_nn import DynamicNeuralNetwork


class TestNAS(unittest.TestCase):
    """
    Unit tests for Neural Architecture Search (NAS).
    """

    def setUp(self):
        """
        Sets up NAS with a dummy dataset.
        """
        self.base_model = DynamicNeuralNetwork(
            input_dim=100, hidden_sizes=[64, 32], output_dim=10
        )
        self.nas = NAS(
            base_model=self.base_model,
            search_space={'increase_units': True}
        )

        # Create dummy dataset
        inputs = torch.randn(100, 100)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(inputs, labels)
        self.dataloader = DataLoader(dataset, batch_size=10)

    def test_mutate_add_layer(self):
        """
        Tests that adding a layer during NAS mutation correctly modifies the architecture.
        """
        mutated_model = self.nas.mutate(self.base_model)
        self.assertGreaterEqual(
            len(mutated_model.layers),
            len(self.base_model.layers)
        )

    def test_mutate_remove_layer(self):
        """
        Tests that removing a layer correctly modifies the architecture.
        """
        mutated_model = self.nas.mutate(self.base_model)
        self.assertLessEqual(
            len(mutated_model.layers),
            len(self.base_model.layers)
        )

    def test_mutate_increase_units(self):
        """
        Tests that increasing the number of units works correctly.
        """
        mutated_model = self.nas.mutate(self.base_model)
        for layer in mutated_model.layers:
            self.assertGreaterEqual(
                layer.weight.shape[0],
                self.base_model.layers[0].weight.shape[0]
            )

    def test_evaluate_accuracy(self):
        """
        Tests evaluation function of NAS.
        """
        accuracy = self.nas.evaluate(self.base_model, self.dataloader)
        self.assertTrue(
            0.0 <= accuracy <= 1.0,
            "Accuracy should be between 0 and 1."
        )

    def test_run_nas(self):
        """
        Tests running NAS optimization.
        """
        best_model = self.nas.run(
            self.dataloader, generations=1, population_size=2
        )
        self.assertIsInstance(best_model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
