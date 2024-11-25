# scripts/visualize.py

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scripts.utils import setup_logging
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Training Metrics")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_training_metrics(metrics, epoch, output_dir='visualizations/training_plots/'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epoch'], y=metrics['loss'])
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, f'training_loss_epoch_{epoch}.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epoch'], y=metrics['accuracy'])
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, f'training_accuracy_epoch_{epoch}.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging('logs/visualize.json.log')
    logger.info("Starting visualization of training metrics...")
    
    # Load metrics history (assuming it's saved as a JSON or similar)
    # For illustration, using dummy data
    metrics_history = {
        'epoch': list(range(1, config['training']['epochs'] + 1)),
        'loss': [0.9, 0.8, 0.7, 0.6, 0.5] * (config['training']['epochs'] // 5),
        'accuracy': [0.6, 0.65, 0.7, 0.75, 0.8] * (config['training']['epochs'] // 5)
    }
    
    # Plot metrics
    plot_training_metrics(metrics_history, epoch=config['training']['epochs'], output_dir='visualizations/training_plots/')
    logger.info("Visualization of training metrics completed.")

if __name__ == "__main__":
    main()
