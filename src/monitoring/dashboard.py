import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd
from datetime import datetime

class DashboardManager:
    def __init__(self):
        self.metrics_history = []
        self.architecture_changes = []
        
    def update_metrics(self, metrics: Dict[str, float]):
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)
        
    def record_architecture_change(self, change_info: Dict):
        change_info['timestamp'] = datetime.now()
        self.architecture_changes.append(change_info)
        
    def render_dashboard(self):
        st.title("Dynamic Neural Network Evolution Monitor")
        
        # Performance Metrics
        st.header("Performance Metrics")
        self._plot_performance_trends()
        
        # Architecture Evolution
        st.header("Architecture Evolution")
        self._plot_architecture_changes()
        
        # Resource Usage
        st.header("Resource Utilization")
        self._plot_resource_usage()
        
    def _plot_performance_trends(self):
        df = pd.DataFrame(self.metrics_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['accuracy'],
            name='Accuracy'
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['loss'],
            name='Loss'
        ))
        
        st.plotly_chart(fig)
