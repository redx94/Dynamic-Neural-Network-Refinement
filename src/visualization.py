# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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
