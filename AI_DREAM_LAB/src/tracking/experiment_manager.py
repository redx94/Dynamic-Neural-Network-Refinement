import mlflow
import torch
import wandb
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

@dataclass
class ExperimentConfig:
    name: str
    tags: Dict[str, str]
    metrics_config: Dict[str, Dict[str, Any]]
    artifacts_config: Dict[str, str]
    tracking_backends: List[str]

class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.active_experiment = None
        self._initialize_tracking_backends()
        
    def track_experiment(self, 
                        model: torch.nn.Module,
                        metrics: Dict[str, float],
                        artifacts: Optional[Dict[str, Any]] = None):
        # Track with MLflow
        with mlflow.start_run(experiment_id=self.active_experiment):
            # Log model architecture evolution
            self._log_architecture_changes(model)
            
            # Track performance metrics
            self._log_metrics(metrics)
            
            # Save artifacts and visualizations
            if artifacts:
                self._store_artifacts(artifacts)
                
            # Generate experiment summary
            summary = self._generate_experiment_summary()
            mlflow.log_dict(summary, "summary.json")
            
    def _log_architecture_changes(self, model: torch.nn.Module):
        architecture_diff = self._compute_architecture_diff(model)
        if architecture_diff:
            mlflow.log_dict(architecture_diff, "architecture_changes.json")
            
            # Log to W&B if enabled
            if "wandb" in self.config.tracking_backends:
                wandb.log({
                    "architecture_changes": architecture_diff,
                    "model_graph": wandb.Graph(model)
                })

    def _generate_experiment_summary(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": self._compute_metrics_summary(),
            "architecture_evolution": self._get_architecture_history(),
            "performance_insights": self._generate_insights()
        }
