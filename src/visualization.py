import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_metrics(metrics, epoch, output_dir="visualizations/training_plots/"):
    """
    Plots training loss and accuracy over epochs.

    Args:
        metrics (dict): Training and validation metrics.
        epoch (int): Current training epoch.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epoch'], y=metrics['loss'])
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, f"training_loss_epoch_{epoch}.png"))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=metrics['epoch'], y=metrics['accuracy'])
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(output_dir, f"training_accuracy_epoch_{epoch}.png"))
    plt.close()
