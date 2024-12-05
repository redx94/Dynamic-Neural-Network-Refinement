import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns
import numpy as np
import io
import base64

class Visualizer:
    @staticmethod
    def plot_learning_curves(train_loss: List[float], 
                           val_loss: List[float]) -> str:
        """Generate learning curves plot and return as base64 string"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    @staticmethod
    def plot_network_structure(model_layers: List[int]) -> str:
        """Visualize network structure"""
        plt.figure(figsize=(12, 6))
        
        # Plot layers as circles
        layer_positions = np.linspace(0, 1, len(model_layers))
        max_neurons = max(model_layers)
        
        for i, layer_size in enumerate(model_layers):
            # Plot neurons
            neuron_positions = np.linspace(0.2, 0.8, layer_size)
            plt.scatter([layer_positions[i]] * layer_size, 
                       neuron_positions,
                       s=100, alpha=0.6)
            
            # Draw connections to next layer
            if i < len(model_layers) - 1:
                next_layer_size = model_layers[i + 1]
                next_positions = np.linspace(0.2, 0.8, next_layer_size)
                
                for j in range(layer_size):
                    for k in range(next_layer_size):
                        plt.plot([layer_positions[i], layer_positions[i + 1]],
                               [neuron_positions[j], next_positions[k]],
                               'gray', alpha=0.1)
        
        plt.title('Network Architecture')
        plt.axis('off')
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    @staticmethod
    def plot_layer_metrics(complexity: List[float], 
                          sparsity: List[float]) -> str:
        """Visualize layer-wise metrics"""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(complexity))
        width = 0.35
        
        plt.bar(x - width/2, complexity, width, label='Complexity')
        plt.bar(x + width/2, sparsity, width, label='Sparsity')
        
        plt.xlabel('Layer')
        plt.ylabel('Metric Value')
        plt.title('Layer-wise Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
