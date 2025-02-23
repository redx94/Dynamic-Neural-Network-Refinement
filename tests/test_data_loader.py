
import unittest
from src.utils.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_load_mnist(self):
        loader = DataLoader()
        train_data = loader.load_mnist(train=True)
        self.assertIsNotNone(train_data)
        
    def test_batch_size(self):
        loader = DataLoader()
        train_data = loader.load_mnist(train=True, batch_size=32)
        for batch in train_data:
            self.assertEqual(batch[0].size(0), 32)
            break
