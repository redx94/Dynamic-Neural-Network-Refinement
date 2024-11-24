import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_metrics(metrics, epoch):
    plt.figure(figsize=(10,5))
    plt.plot(metrics['loss'], label='Loss')
    plt.plot(metrics['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(f'visualizations/training_metrics_epoch_{epoch}.png')
    plt.close()

def plot_threshold_evolution(thresholds, epoch):
    for key, values in thresholds.items():
        plt.figure()
        plt.plot(values['simple'], label='Simple')
        plt.plot(values['moderate'], label='Moderate')
        plt.xlabel('Epoch')
        plt.ylabel(f'{key.capitalize()} Threshold')
        plt.title(f'{key.capitalize()} Threshold Evolution')
        plt.legend()
        plt.savefig(f'visualizations/{key}_threshold_evolution_epoch_{epoch}.png')
        plt.close()
