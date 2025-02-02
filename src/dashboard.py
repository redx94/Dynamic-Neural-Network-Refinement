import streamlit as st
import plotly.graph_objects as go
import numpy as np


class Dashboard:
    """
    Streamlit-based dashboard for visualizing neural network training progress 
    and performance metrics.
    """

    def __init__(self):
        st.set_page_config(layout="wide", page_title="Neural Network Training Dashboard")
        self.initialize_session_state()

    def initialize_session_state(self):
        """
        Initializes session state variables if they are not already present.
        """
        if "metrics" not in st.session_state:
            st.session_state.metrics = {
                "training_loss": [],
                "validation_loss": [],
                "accuracy": [],
                "complexity_metrics": [],
                "gradient_norms": [],
                "learning_rates": [],
                "batch_sizes": [],
                "memory_usage": []
            }

    def render(self):
        """
        Renders the Streamlit dashboard UI.
        """
        st.title("Neural Network Training Dashboard")
        st.markdown("### Monitor your model's training progress in real-time.")

        # Metrics Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            self.metric_card(
                "Model Accuracy",
                f"{self.get_latest_metric('accuracy'):.2f}%",
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

        # Main dashboard sections
        tab1, tab2, tab3 = st.tabs([
            "📊 Training Progress",
            "🔍 Model Analysis",
            "📈 System Statistics"
        ])

        with tab1:
            st.markdown("### Training Progress")
            self.render_training_metrics()

        with tab2:
            st.markdown("### Model Analysis")
            self.render_model_analysis()

        with tab3:
            st.markdown("### System Statistics")
            self.render_system_metrics()

    def metric_card(self, title, value, helper_text=""):
        """
        Renders a metric card with title, value, and description.

        Args:
            title (str): Title of the metric.
            value (str): Metric value.
            helper_text (str): Additional description.
        """
        st.markdown(f"""
        <div style="background-color:#1c1c1c; padding:20px; border-radius:10px;">
            <h3 style="color:#00f9f9;">{title}</h3>
            <h2 style="color:#ffffff;">{value}</h2>
            <div class="helper-text">{helper_text}</div>
        </div>
        """, unsafe_allow_html=True)

    def get_latest_metric(self, metric_name):
        """
        Retrieves the most recent metric value.

        Args:
            metric_name (str): Name of the metric.

        Returns:
            float: Latest metric value.
        """
        values = st.session_state.metrics.get(metric_name, [])
        return values[-1] if values else 0.0

    def render_training_metrics(self):
        """
        Renders training and validation loss curves.
        """
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Loss Curves")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics["training_loss"],
                name="Training Loss",
                line=dict(color="cyan")
            ))
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics["validation_loss"],
                name="Validation Loss",
                line=dict(color="magenta")
            ))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Accuracy Progression")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.metrics["accuracy"],
                name="Accuracy",
                fill="tozeroy",
                line=dict(color="yellow")
            ))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_model_analysis(self):
        """
        Renders architectural analysis and gradient norms.
        """
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Gradient Flow")
            if st.session_state.metrics["gradient_norms"]:
                fig = go.Figure(data=go.Heatmap(
                    z=np.array(st.session_state.metrics["gradient_norms"]),
                    colorscale="Viridis"
                ))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Learning Rate Schedule")
            if st.session_state.metrics["learning_rates"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics["learning_rates"],
                    name="Learning Rate",
                    line=dict(color="cyan")
                ))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

    def render_system_metrics(self):
        """
        Renders system memory usage.
        """
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Memory Usage (MB)")
            if st.session_state.metrics["memory_usage"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics["memory_usage"],
                    name="Memory Usage",
                    line=dict(color="orange")
                ))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.render()
