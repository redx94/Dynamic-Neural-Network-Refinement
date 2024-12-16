
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MonitoringAndVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(self.model)
        self.metrics_history = {'loss': [], 'accuracy': [], 'complexity': []}
        
    def compute_feature_attributions(self, data, complexities):
        attributions = {}
        for i in range(data.size(0)):
            input = data[i].unsqueeze(0).requires_grad_(True).to(self.device)
            complexity = complexities[i].unsqueeze(0).to(self.device)
            attribution, _ = self.ig.attribute(input, target=0, return_convergence_delta=True)
            attributions[i] = attribution.cpu().detach().numpy()
        return attributions
        
    def track_metrics(self, loss, accuracy, complexity):
        self.metrics_history['loss'].append(loss)
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['complexity'].append(complexity)
        
    def create_dashboard(self):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Over Time', 'Accuracy Over Time', 
                          'Complexity Distribution', 'Feature Importance')
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(y=self.metrics_history['loss'], name='Loss'),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(y=self.metrics_history['accuracy'], name='Accuracy'),
            row=1, col=2
        )
        
        # Complexity distribution
        fig.add_trace(
            go.Histogram(x=self.metrics_history['complexity'], name='Complexity'),
            row=2, col=1
        )
        
        # Feature importance plot
        if hasattr(self, 'last_attribution'):
            fig.add_trace(
                go.Bar(y=self.last_attribution.mean(axis=0), name='Feature Importance'),
                row=2, col=2
            )
            
        fig.update_layout(height=800, title_text="Model Monitoring Dashboard")
        return fig
        
    def save_dashboard(self, filename='dashboard.html'):
        fig = self.create_dashboard()
        fig.write_html(filename)
