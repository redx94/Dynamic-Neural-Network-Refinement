
import unittest
from src.utils.data_loader import get_data_loaders

class TestDataLoader(unittest.TestCase):
    def test_load_mnist(self):
        train_loader, val_loader = get_data_loaders()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
    def test_batch_size(self):
        train_loader, val_loader = get_data_loaders(batch_size=32)
        for batch in train_loader:
            self.assertEqual(batch[0].size(0), 32)
            break
