import unittest
import subprocess
import os
import yaml

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure the necessary directories exist
        os.makedirs('models/final/', exist_ok=True)
        os.makedirs('logs/', exist_ok=True)
    
    def test_training_pipeline(self):
        # Load configurations
        with open('config/train_config.yaml', 'r') as file:
            train_config = yaml.safe_load(file)
        
        # Test just 1 epoch for integration test by overriding yaml temporarily
        # But instead of rewriting yaml, we can modify src.train or just let it train on the tiny set
        # Actually src.train.py defaults to 2 epochs for quick testing.
        
        # Run training
        train_result = subprocess.run(['python', '-m', 'src.train'], capture_output=True)
        self.assertEqual(train_result.returncode, 0, f"Training failed with error: {train_result.stderr.decode()}")
        
        # Check if final model exists
        final_model_path = train_config['output'].get('final_model_path', 'best_model.pth')
        self.assertTrue(os.path.exists(final_model_path), "Final model not found after training.")

if __name__ == '__main__':
    unittest.main()
