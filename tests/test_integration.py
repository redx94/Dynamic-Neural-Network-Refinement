# tests/test_integration.py

import unittest
import subprocess
import os
import yaml

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure the necessary directories exist
        os.makedirs('models/final/', exist_ok=True)
        os.makedirs('logs/', exist_ok=True)
    
    def test_training_and_evaluation_pipeline(self):
        # Load configurations
        with open('config/train_config.yaml', 'r') as file:
            train_config = yaml.safe_load(file)
        with open('config/eval_config.yaml', 'r') as file:
            eval_config = yaml.safe_load(file)
        
        # Run training
        train_result = subprocess.run(['python', '-m', 'scripts.train', '--config', 'config/train_config.yaml'], capture_output=True)
        self.assertEqual(train_result.returncode, 0, f"Training failed with error: {train_result.stderr.decode()}")
        
        # Check if final model exists
        final_model_path = train_config['output']['final_model_path']
        self.assertTrue(os.path.exists(final_model_path), "Final model not found after training.")
        
        # Run evaluation
        eval_result = subprocess.run(['python', 'scripts/evaluate.py', '--config', 'config/eval_config.yaml', '--model_path', final_model_path], capture_output=True)
        self.assertEqual(eval_result.returncode, 0, f"Evaluation failed with error: {eval_result.stderr.decode()}")
        
        # Check evaluation results
        eval_results_path = eval_config['output']['evaluation_results_path']
        self.assertTrue(os.path.exists(eval_results_path), "Evaluation results not found after evaluation.")
        
        # Optionally, verify contents of evaluation results
        import pandas as pd
        df = pd.read_csv(eval_results_path)
        self.assertFalse(df.empty, "Evaluation results are empty.")

if __name__ == '__main__':
    unittest.main()
