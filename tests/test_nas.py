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
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
