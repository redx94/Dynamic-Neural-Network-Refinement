import unittest
import torch
import numpy as np
from src.models.neural_network import DynamicNeuralNetwork, DynamicLayer
from src.models.refinement import NetworkRefinement
from src.training.trainer import Trainer
from src.data.dataset import generate_synthetic_data, create_data_loaders

class TestDynamicNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.input_size = 10
        self.output_size = 1
        self.hidden_sizes = [64, 32]
        self.model = DynamicNeuralNetwork(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=self.hidden_sizes
        )
        
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        # Check input/output sizes
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.output_size, self.output_size)
        
        # Check layer structure
        self.assertEqual(len(self.model.layers), len(self.hidden_sizes))
        self.assertEqual(
            self.model.layers[0].linear.in_features, 
            self.input_size
        )
        self.assertEqual(
            self.model.output_layer.out_features, 
            self.output_size
        )
        
    def test_forward_pass(self):
        """Test forward pass with sample input"""
        batch_size = 32
        x = torch.randn(batch_size, self.input_size)
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(
            output.shape, 
            (batch_size, self.output_size)
        )
        
    def test_add_layer(self):
        """Test dynamic layer addition"""
        initial_layers = len(self.model.layers)
        new_size = 16
        
        self.model.add_layer(new_size)
        
        # Check if layer was added
        self.assertEqual(len(self.model.layers), initial_layers + 1)
        self.assertEqual(
            self.model.layers[-1].linear.out_features,
            new_size
        )
        
    def test_remove_layer(self):
        """Test layer removal"""
        initial_layers = len(self.model.layers)
        
        self.model.remove_layer()
        
        # Check if layer was removed
        self.assertEqual(len(self.model.layers), initial_layers - 1)
        
    def test_adjust_layer_width(self):
        """Test layer width adjustment"""
        layer_idx = 0
        new_width = 48
        
        original_width = self.model.layers[layer_idx].linear.out_features
        self.model.adjust_layer_width(layer_idx, new_width)
        
        # Check if width was adjusted
        self.assertEqual(
            self.model.layers[layer_idx].linear.out_features,
            new_width
        )
        self.assertNotEqual(
            self.model.layers[layer_idx].linear.out_features,
            original_width
        )

class TestNetworkRefinement(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.model = DynamicNeuralNetwork(
            input_size=10,
            output_size=1
        )
        self.refinement = NetworkRefinement(self.model)
        
    def test_complexity_calculation(self):
        """Test layer complexity calculation"""
        complexity = self.refinement.calculate_layer_complexity(0)
        
        # Check if complexity is within valid range
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
        
    def test_sparsity_calculation(self):
        """Test layer sparsity calculation"""
        sparsity = self.refinement.calculate_layer_sparsity(0)
        
        # Check if sparsity is within valid range
        self.assertGreaterEqual(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)
        
    def test_network_analysis(self):
        """Test network analysis"""
        metrics = self.refinement.analyze_network()
        
        # Check if metrics contain required keys
        self.assertIn('complexity', metrics)
        self.assertIn('sparsity', metrics)
        
        # Check if metrics lists have correct length
        self.assertEqual(
            len(metrics['complexity']),
            len(self.model.layers)
        )

class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.model = DynamicNeuralNetwork(
            input_size=10,
            output_size=1
        )
        self.trainer = Trainer(self.model)
        
        # Generate synthetic data
        X, y = generate_synthetic_data(n_samples=100)
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        self.train_loader, self.val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )
        
    def test_training_loop(self):
        """Test basic training functionality"""
        history = self.trainer.train(
            self.train_loader,
            self.val_loader,
            epochs=2,
            refinement_frequency=1
        )
        
        # Check if history contains required metrics
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('learning_rate', history)
        
        # Check if losses are decreasing
        self.assertGreaterEqual(
            history['train_loss'][0],
            history['train_loss'][-1]
        )
        
    def test_learning_curves(self):
        """Test learning curves generation"""
        self.trainer.train(
            self.train_loader,
            self.val_loader,
            epochs=2
        )
        train_loss, val_loss = self.trainer.get_learning_curves()
        
        # Check if curves have correct length
        self.assertEqual(len(train_loss), 2)
        self.assertEqual(len(val_loss), 2)

if __name__ == '__main__':
    unittest.main()
