
import streamlit as st
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import numpy as np
import torch
from datetime import datetime
import plotly.express as px

class Dashboard:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Neural Network Training Dashboard")
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
                .helper-text {{
                    font-size: 14px;
                    color: #888;
                    margin-top: 5px;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("Neural Network Training Dashboard")
        st.markdown("*Monitor your model's training progress in real-time*")

        # Top metrics row with explanations
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.metric_card(
                "Model Accuracy", 
                f"{self.get_latest_metric('accuracy'):.2%}",
                "How well the model performs on test data"
            )
        with col2:
            self.metric_card(
                "Training Loss", 
                f"{self.get_latest_metric('training_loss'):.4f}",
                "Error rate during training (lower is better)"
            )
        with col3:
            self.metric_card(
                "Validation Loss", 
                f"{self.get_latest_metric('validation_loss'):.4f}",
                "Error rate on validation data"
            )
        with col4:
            self.metric_card(
                "Memory Usage", 
                f"{self.get_latest_metric('memory_usage'):.1f} MB",
                "Current GPU/CPU memory consumption"
            )

        # Main dashboard tabs with descriptions
        tab1, tab2, tab3 = st.tabs([
            "📈 Training Progress", 
            "🎯 Model Analysis", 
            "⚙️ System Stats"
        ])

        with tab1:
            st.markdown("### Training Progress")
            st.markdown("*Track how your model improves over time*")
            self.render_training_metrics()

        with tab2:
            st.markdown("### Model Analysis")
            st.markdown("*Analyze internal model behavior and patterns*")
            self.render_model_analysis()

        with tab3:
            st.markdown("### System Statistics")
            st.markdown("*Monitor system resource usage*")
            self.render_system_metrics()

    def metric_card(self, title, value, helper_text=""):
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color: {self.theme['primary']};">{title}</h3>
                <h2 style="color: {self.theme['text']};">{value}</h2>
                <div class="helper-text">{helper_text}</div>
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
            st.markdown("#### Loss Curves")
            st.markdown("*Compare training and validation loss*")
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
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Accuracy Progression")
            st.markdown("*Model accuracy over time*")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['accuracy'],
                name='Accuracy',
                fill='tozeroy',
                line=dict(color=self.theme['primary'])
            ))
            fig.update_layout(
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_model_analysis(self):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Gradient Flow")
            st.markdown("*How gradients propagate through layers*")
            if st.session_state.metrics['gradient_norms']:
                fig = go.Figure(data=go.Heatmap(
                    z=np.array(st.session_state.metrics['gradient_norms']),
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Layer Activations")
            st.markdown("*Distribution of neuron activations*")
            if st.session_state.metrics['layer_activations']:
                fig = go.Figure(data=go.Violin(
                    y=np.array(st.session_state.metrics['layer_activations']).flatten(),
                    box_visible=True,
                    line_color=self.theme['primary']
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_system_metrics(self):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Learning Rate")
            st.markdown("*Learning rate adaptation over time*")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['learning_rates'],
                name='Learning Rate',
                line=dict(color=self.theme['primary'])
            ))
            fig.update_layout(
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Memory Usage")
            st.markdown("*System memory consumption*")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics['memory_usage'],
                name='Memory Usage (MB)',
                line=dict(color=self.theme['secondary'])
            ))
            fig.update_layout(
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.render()
