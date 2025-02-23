import unittest
import subprocess
import os
import yaml


class TestE2E(unittest.TestCase):
    """
    End-to-end test suite for training, evaluating, and deploying the model.
    """

    def setUp(self):
        """
        Sets up necessary directories for testing.
        """
        os.makedirs("data/raw/", exist_ok=True)
        os.makedirs("models/final/", exist_ok=True)
        os.makedirs("data/synthetic/", exist_ok=True)
        os.makedirs("logs/", exist_ok=True)

    def test_full_pipeline(self):
        """
        Executes the full training, evaluation, pruning, and quantization pipeline.
        """

        # Step 1: Train the model
        train_result = subprocess.run(
            [
                "python", "scripts/train.py", "--config",
                "config/train_config.yaml"
            ],
            capture_output=True
        )
        self.assertEqual(
            train_result.returncode, 0,
            f"Training failed: {train_result.stderr.decode()}"
        )

        # Step 2: Evaluate the model
        with open("config/train_config.yaml") as config_file:
            train_config = yaml.safe_load(config_file)
        model_path = train_config["output"]["final_model_path"]

        eval_result = subprocess.run(
            [
                "python", "scripts/evaluate.py", "--config",
                "config/eval_config.yaml", "--model_path", model_path
            ],
            capture_output=True
        )
        self.assertEqual(
            eval_result.returncode, 0,
            f"Evaluation failed: {eval_result.stderr.decode()}"
        )

        # Step 3: Apply Pruning
        pruned_model_path = "models/pruned/pruned_model.pth"
        prune_result = subprocess.run(
            [
                "python", "scripts/prune.py", "--config",
                "config/train_config.yaml", "--model_path", model_path
            ],
            capture_output=True
        )
        self.assertEqual(
            prune_result.returncode, 0,
            f"Pruning failed: {prune_result.stderr.decode()}"
        )
        self.assertTrue(os.path.exists(pruned_model_path), "Pruned model not found.")

        # Step 4: Apply Quantization
        quantized_model_path = "models/quantized/quantized_model.pth"
        quantize_result = subprocess.run(
            [
                "python", "scripts/quantize.py", "--config",
                "config/train_config.yaml", "--model_path", pruned_model_path
            ],
            capture_output=True
        )
        self.assertEqual(
            quantize_result.returncode, 0,
            f"Quantization failed: {quantize_result.stderr.decode()}"
        )
        self.assertTrue(
            os.path.exists(quantized_model_path),
            "Quantized model not found."
        )

        # Step 5: Generate Synthetic Data
        synthetic_data_path = "data/synthetic/synthetic_sample.pth"
        generate_result = subprocess.run(
            [
                "python", "scripts/generate_synthetic.py", "--config",
                "config/train_config.yaml", "--model_path", quantized_model_path
            ],
            capture_output=True
        )
        self.assertEqual(
            generate_result.returncode, 0,
            f"Synthetic data generation failed: {generate_result.stderr.decode()}"
        )
        self.assertTrue(
            os.path.exists(synthetic_data_path),
            "Synthetic sample data not found."
        )

        # Step 6: Run Visualization
        visualize_result = subprocess.run(
            [
                "python", "scripts/visualize.py", "--config",
                "config/train_config.yaml"
            ],
            capture_output=True
        )
        self.assertEqual(
            visualize_result.returncode, 0,
            f"Visualization failed: {visualize_result.stderr.decode()}"
        )

        # Check if visualization files are created
        training_plot = "visualizations/training_plots/training_loss_epoch_20.png"
        self.assertTrue(
            os.path.exists(training_plot),
            "Training visualization plot not found."
        )


if __name__ == "__main__":
    unittest.main()
