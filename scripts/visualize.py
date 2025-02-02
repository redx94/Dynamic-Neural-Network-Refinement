import argparse
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    """
    Parses command-line arguments for visualizing training metrics.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize Training Metrics")
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)
    return parser.parse_args()


def load_config(config_path):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def plot_training_metrics(metrics, output_dir="visualizations/training_plots/"):
    """
    Plots training loss and accuracy over epochs.

    Args:
        metrics (dict): Training and validation metrics.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epochs'], y=metrics['loss'])
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epochs'], y=metrics['accuracy'])
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(output_dir, "training_accuracy.png"))
    plt.close()


def main():
    """
    Main function to visualize training metrics.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Example training metrics (Replace with actual logs)
    metrics_history = {
        'epochs': list(range(1, config['training']['epochs'] + 1)),
        'loss': [0.9, 0.8, 0.7, 0.6, 0.5],
        'accuracy': [0.6, 0.65, 0.7, 0.75, 0.8]
    }

    # Plot metrics
    plot_training_metrics(metrics_history)
    print("Visualization of training metrics completed.")


if __name__ == "__main__":
    main()
