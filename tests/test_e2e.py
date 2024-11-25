# tests/test_e2e.py

import unittest
import subprocess
import os
import yaml

class TestE2E(unittest.TestCase):
    def setUp(self):
        # Ensure necessary directories exist
        os.makedirs('data/raw/', exist_ok=True)
        os.makedirs('models/final/', exist_ok=True)
        os.makedirs('data/synthetic/', exist_ok=True)
        os.makedirs('logs/', exist_ok=True)
    
    def test_full_pipeline(self):
        # Step 1: Data Preparation (Assuming you have a script or manual process)
        # For illustration, we'll skip actual data preparation
        
        # Step 2: Train the model
        train_result = subprocess.run(['python', 'scripts/train.py', '--config', 'config/train_config.yaml'], capture_output=True)
        self.assertEqual(train_result.returncode, 0, f"Training failed with error: {train_result.stderr.decode()}")
        
        # Step 3: Evaluate the model
        train_config = yaml.safe_load(open('config/train_config.yaml'))
        final_model_path = train_config['output']['final_model_path']
        eval_config = yaml.safe_load(open('config/eval_config.yaml'))
        eval_result = subprocess.run(['python', 'scripts/evaluate.py', '--config', 'config/eval_config.yaml', '--model_path', final_model_path], capture_output=True)
        self.assertEqual(eval_result.returncode, 0, f"Evaluation failed with error: {eval_result.stderr.decode()}")
        
        # Step 4: Apply Pruning
        prune_result = subprocess.run(['python', 'scripts/prune.py', '--config', 'config/train_config.yaml', '--model_path', final_model_path], capture_output=True)
        self.assertEqual(prune_result.returncode, 0, f"Pruning failed with error: {prune_result.stderr.decode()}")
        pruned_model_path = 'models/pruned/pruned_model.pth'
        self.assertTrue(os.path.exists(pruned_model_path), "Pruned model not found.")
        
        # Step 5: Apply Quantization
        quantize_result = subprocess.run(['python', 'scripts/quantize.py', '--config', 'config/train_config.yaml', '--model_path', pruned_model_path], capture_output=True)
        self.assertEqual(quantize_result.returncode, 0, f"Quantization failed with error: {quantize_result.stderr.decode()}")
        quantized_model_path = 'models/quantized/quantized_model.pth'
        self.assertTrue(os.path.exists(quantized_model_path), "Quantized model not found.")
        
        # Step 6: Generate Synthetic Data
        generate_result = subprocess.run(['python', 'scripts/generate_synthetic.py', '--config', 'config/train_config.yaml', '--model_path', quantized_model_path], capture_output=True)
        self.assertEqual(generate_result.returncode, 0, f"Synthetic data generation failed with error: {generate_result.stderr.decode()}")
        synthetic_data_simple = 'data/synthetic/synthetic_simple.pth'
        self.assertTrue(os.path.exists(synthetic_data_simple), "Synthetic simple data not found.")
        
        # Step 7: Run Visualization
        visualize_result = subprocess.run(['python', 'scripts/visualize.py', '--config', 'config/train_config.yaml'], capture_output=True)
        self.assertEqual(visualize_result.returncode, 0, f"Visualization failed with error: {visualize_result.stderr.decode()}")
        # Check if visualization files are created
        training_plot = 'visualizations/training_plots/training_loss_epoch_20.png'
        self.assertTrue(os.path.exists(training_plot), "Training visualization plot not found.")

if __name__ == '__main__':
    unittest.main()
