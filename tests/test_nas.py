# tests/test_nas.py

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from src.nas import NAS

class TestNAS(unittest.TestCase):
    def setUp(self):
        # Define initial thresholds
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        self.hybrid_thresholds = HybridThresholds(
            initial_thresholds=initial_thresholds,
            annealing_start_epoch=5,
            total_epochs=20
        )
        
        # Initialize base model
        self.base_model = DynamicNeuralNetwork(hybrid_thresholds=self.hybrid_thresholds)
        
        # Define search space
        self.search_space = {
            'add_layer': True,
            'remove_layer': True,
            'increase_units': True,
            'decrease_units': True
        }
        
        # Initialize NAS
        self.nas = NAS(
            base_model=self.base_model,
            search_space=self.search_space,
            device='cpu'
        )
        
        # Create dummy dataset
        dummy_input = torch.randn(100, 100)  # Assuming input size of 100
        dummy_labels = torch.randint(0, 10, (100,))  # Assuming 10 classes
        dataset = TensorDataset(dummy_input, dummy_labels)
        self.dataloader = DataLoader(dataset, batch_size=10)
    
    def test_mutate_add_layer(self):
        # Test adding a layer
        model_before = self.nas.base_model
        mutated_model = self.nas.mutate(model_before)
        
        # Check if layer3 has been modified to include an additional layer
        if hasattr(model_before, 'layer3'):
            if isinstance(model_before.layer3, torch.nn.Sequential):
                self.assertEqual(len(mutated_model.layer3), len(model_before.layer3) + 1)
            else:
                self.assertTrue(isinstance(mutated_model.layer3, torch.nn.Sequential))
                self.assertEqual(len(mutated_model.layer3), 2)  # Original + new layer
        else:
            self.assertTrue(hasattr(mutated_model, 'layer3'))
            self.assertTrue(isinstance(mutated_model.layer3, torch.nn.Sequential))
            self.assertEqual(len(mutated_model.layer3), 1)  # New layer added
    
    def test_mutate_remove_layer(self):
        # Ensure the base model has layer3
        self.assertTrue(hasattr(self.base_model, 'layer3'))
        
        # Test removing a layer
        mutated_model = self.nas.mutate(self.base_model)
        
        # Check if layer3 has been removed
        self.assertFalse(hasattr(mutated_model, 'layer3'))
    
    def test_mutate_increase_units(self):
        # Test increasing units in layer1
        mutated_model = self.nas.mutate(self.base_model)
        
        # Check if layer1 has increased units
        original_units = self.base_model.layer1.layer.out_features
        mutated_units = mutated_model.layer1.layer.out_features
        self.assertEqual(mutated_units, original_units * 2)  # Assuming doubling units
    
    def test_mutate_decrease_units(self):
        # Test decreasing units in layer1
        mutated_model = self.nas.mutate(self.base_model)
        
        # Check if layer1 has decreased units
        original_units = self.base_model.layer1.layer.out_features
        mutated_units = mutated_model.layer1.layer.out_features
        self.assertEqual(mutated_units, original_units // 2)  # Assuming halving units
    
    def test_evaluate_accuracy(self):
        # Test evaluation function
        accuracy = self.nas.evaluate(self.base_model, self.dataloader)
        self.assertTrue(0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1.")
    
    def test_run_nas(self):
        # Test the NAS run function
        best_model = self.nas.run(self.dataloader, generations=1, population_size=2)
        self.assertIsInstance(best_model, torch.nn.Module)
        # Optionally, check if best_model has better or equal accuracy than base_model
        base_accuracy = self.nas.evaluate(self.base_model, self.dataloader)
        best_accuracy = self.nas.evaluate(best_model, self.dataloader)
        self.assertTrue(best_accuracy >= base_accuracy, "Best model should have accuracy >= base model.")
    
    def test_run_nas_multiple_generations(self):
        # Test NAS with multiple generations
        best_model = self.nas.run(self.dataloader, generations=3, population_size=2)
        self.assertIsInstance(best_model, torch.nn.Module)
        # Further checks can be added based on specific mutation strategies
    
    def test_run_nas_no_improvement(self):
        # Test NAS when no improvement is expected
        # For deterministic behavior, override random choices
        original_mutate = self.nas.mutate
        def mock_mutate(model):
            return copy.deepcopy(model)  # No mutation
        self.nas.mutate = mock_mutate
        
        best_model = self.nas.run(self.dataloader, generations=2, population_size=2)
        self.assertIsInstance(best_model, torch.nn.Module)
        # Since no mutation, best_model should be the same as base_model
        self.assertTrue(torch.equal(
            best_model.layer1.layer.weight, 
            self.base_model.layer1.layer.weight
        ), "Best model should be identical to base model when no mutations occur.")
        
        # Restore original mutate function
        self.nas.mutate = original_mutate

if __name__ == '__main__':
    unittest.main()
