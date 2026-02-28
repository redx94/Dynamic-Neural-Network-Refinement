import unittest
import subprocess
import os
import yaml
import time


class TestE2E(unittest.TestCase):
    """
    End-to-end test suite for training and serving the model.
    """

    def setUp(self):
        os.makedirs("data/raw/", exist_ok=True)
        os.makedirs("models/final/", exist_ok=True)
        os.makedirs("logs/", exist_ok=True)

    def test_full_pipeline(self):
        """
        Executes the basic training and inference pipeline end-to-end.
        """
        # Step 1: Train the model
        train_result = subprocess.run(
            ["python", "-m", "src.train"],
            capture_output=True,
            cwd="."
        )
        self.assertEqual(
            train_result.returncode, 0,
            f"Training failed: {train_result.stderr.decode()}"
        )

        with open("config/train_config.yaml") as config_file:
            train_config = yaml.safe_load(config_file)
        model_path = train_config["output"]["final_model_path"]
        self.assertTrue(os.path.exists(model_path), "Trained model not found.")

        # Optional: Step 2: Start API and send request if we want a true E2E 
        # For CI we will run a dummy inference to test the model locally
        from src.model import DynamicNeuralNetwork
        from src.analyzer import Analyzer
        import torch
        
        device = 'cpu'
        model = DynamicNeuralNetwork(hybrid_thresholds=None)
        # Load state dict strictly (in case of missing keys, wait app.py deals with it)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        analyzer = Analyzer()
        dummy_input = torch.randn((1, 784))
        complexities = analyzer.analyze(dummy_input)
        output = model(dummy_input, complexities)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
