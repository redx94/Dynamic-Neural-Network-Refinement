
import streamlit as st
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import numpy as np
import torch
from datetime import datetime
import plotly.express as px

class Dashboard:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Neural Network Refinement Hub")
        self.initialize_session_state()
        self.theme = self.initialize_theme()

    def initialize_theme(self):
        return {
            'background': '#0f1116',
            'text': '#ffffff',
            'primary': '#00ff9f',
            'secondary': '#7b2cbf'
        }

    def initialize_session_state(self):
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {
                'training_loss': [],
                'validation_loss': [],
                'accuracy': [],
                'complexity_metrics': [],
                'threshold_values': [],
                'gradient_norms': [],
                'layer_activations': [],
                'learning_rates': [],
                'batch_sizes': [],
                'memory_usage': []
            }

    def render(self):
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {self.theme['background']};
                    color: {self.theme['text']};
                }}
                .metric-card {{
                    background-color: #1c1c1c;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid {self.theme['primary']};
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("🧠 Neural Network Refinement Hub")

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.metric_card("Current Accuracy", 
                           f"{self.get_latest_metric('accuracy'):.2%}")
        with col2:
            self.metric_card("Training Loss", 
                           f"{self.get_latest_metric('training_loss'):.4f}")
        with col3:
            self.metric_card("Validation Loss", 
                           f"{self.get_latest_metric('validation_loss'):.4f}")
        with col4:
            self.metric_card("Memory Usage", 
                           f"{self.get_latest_metric('memory_usage'):.1f} MB")

        # Main dashboard tabs
        tab1, tab2, tab3 = st.tabs(["📈 Training Metrics", "🎯 Model Analysis", "⚙️ System Metrics"])

        with tab1:
            self.render_training_metrics()

        with tab2:
            self.render_model_analysis()

        with tab3:
            self.render_system_metrics()

    def metric_card(self, title, value):
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: {self.theme['primary']};">{title}</h3>
                <h2 style="color: {self.theme['text']};">{value}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    def get_latest_metric(self, metric_name):
        values = st.session_state.metrics.get(metric_name, [])
        return values[-1] if values else 0.0

    def render_training_metrics(self):
        col1, col2 = st.columns(2)

        with col1:
            # Loss curves
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['training_loss'],
                name='Training Loss',
                line=dict(color=self.theme['primary'])
            ))
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['validation_loss'],
                name='Validation Loss',
                line=dict(color=self.theme['secondary'])
            ))
            fig.update_layout(
                title='Loss Curves',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Accuracy progression
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['accuracy'],
                name='Accuracy',
                fill='tozeroy',
                line=dict(color=self.theme['primary'])
            ))
            fig.update_layout(
                title='Accuracy Progression',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_model_analysis(self):
        col1, col2 = st.columns(2)

        with col1:
            # Gradient norms heatmap
            if st.session_state.metrics['gradient_norms']:
                fig = go.Figure(data=go.Heatmap(
                    z=np.array(st.session_state.metrics['gradient_norms']),
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    title='Gradient Flow Analysis',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Layer activations
            if st.session_state.metrics['layer_activations']:
                fig = go.Figure(data=go.Violin(
                    y=np.array(st.session_state.metrics['layer_activations']).flatten(),
                    box_visible=True,
                    line_color=self.theme['primary']
                ))
                fig.update_layout(
                    title='Layer Activation Distribution',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_system_metrics(self):
        col1, col2 = st.columns(2)

        with col1:
            # Learning rate progression
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['learning_rates'],
                name='Learning Rate',
                line=dict(color=self.theme['primary'])
            ))
            fig.update_layout(
                title='Learning Rate Adaptation',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Memory usage
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['memory_usage'],
                name='Memory Usage (MB)',
                line=dict(color=self.theme['secondary'])
            ))
            fig.update_layout(
                title='Memory Usage Over Time',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    def update_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            if key in st.session_state.metrics:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                st.session_state.metrics[key].append(value)

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.render()
