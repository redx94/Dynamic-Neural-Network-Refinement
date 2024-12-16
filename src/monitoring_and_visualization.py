
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

class MonitoringSystem:
    def __init__(self):
        self.metrics = {
            'training_loss': [],
            'validation_loss': [],
            'accuracy': [],
            'complexity_metrics': [],
            'gradient_metrics': [],
            'threshold_values': []
        }
        
    def update_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
    def create_dashboard(self):
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Training Loss', 'Validation Loss',
                'Accuracy', 'Complexity Metrics',
                'Gradient Flow', 'Threshold Values'
            )
        )
        
        # Training Loss
        fig.add_trace(
            go.Scatter(y=self.metrics['training_loss'], name='Training Loss'),
            row=1, col=1
        )
        
        # Validation Loss
        fig.add_trace(
            go.Scatter(y=self.metrics['validation_loss'], name='Validation Loss'),
            row=1, col=2
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(y=self.metrics['accuracy'], name='Accuracy'),
            row=2, col=1
        )
        
        # Complexity Metrics
        if self.metrics['complexity_metrics']:
            complexities = np.array(self.metrics['complexity_metrics'])
            fig.add_trace(
                go.Scatter(y=complexities.mean(axis=1), name='Avg Complexity'),
                row=2, col=2
            )
        
        # Gradient Flow
        if self.metrics['gradient_metrics']:
            gradients = np.array(self.metrics['gradient_metrics'])
            fig.add_trace(
                go.Heatmap(z=gradients, name='Gradient Flow'),
                row=3, col=1
            )
        
        # Threshold Values
        if self.metrics['threshold_values']:
            thresholds = np.array(self.metrics['threshold_values'])
            fig.add_trace(
                go.Scatter(y=thresholds, name='Threshold Values'),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, width=1000, title_text="Model Monitoring Dashboard")
        return fig
    
    def save_dashboard(self, filename=None):
        if filename is None:
            filename = f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        fig = self.create_dashboard()
        fig.write_html(filename)
        return filename

    def log_metrics(self, step, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if key in self.metrics:
                self.metrics[key].append(value)
