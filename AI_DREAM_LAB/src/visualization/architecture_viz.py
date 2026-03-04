import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List
import torch.nn as nn

class ArchitectureVisualizer:
    def __init__(self):
        self.history = []
        
    def capture_architecture(self, model: nn.Module, metrics: Dict[str, float]):
        G = nx.DiGraph()
        layers_info = self._extract_layer_info(model)
        self.history.append({
            'architecture': layers_info,
            'metrics': metrics,
            'timestamp': len(self.history)
        })
        
    def create_interactive_visualization(self) -> go.Figure:
        fig = go.Figure()
        
        for timestamp, snapshot in enumerate(self.history):
            node_trace, edge_trace = self._create_network_traces(snapshot['architecture'])
            
            fig.add_trace(node_trace)
            fig.add_trace(edge_trace)
            
            fig.frames = self._create_animation_frames()
            
        fig.update_layout(
            title='Neural Architecture Evolution',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[self._create_animation_menu()]
        )
        
        return fig
    
    def _extract_layer_info(self, model: nn.Module) -> Dict:
        layer_info = {}
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer_info[name] = {
                    'type': layer.__class__.__name__,
                    'in_features': layer.in_features if hasattr(layer, 'in_features') else layer.in_channels,
                    'out_features': layer.out_features if hasattr(layer, 'out_features') else layer.out_channels
                }
        return layer_info
