import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    Occlusion
)

class ModelExplainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.integrated_gradients = IntegratedGradients(model)
        self.deep_lift = DeepLift(model)
        self.gradient_shap = GradientShap(model)
        
    def explain_prediction(self,
                         input_tensor: torch.Tensor,
                         target_class: Optional[int] = None,
                         method: str = 'integrated_gradients') -> Dict[str, torch.Tensor]:
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients.attribute(
                input_tensor,
                target=target_class,
                n_steps=200
            )
        elif method == 'deep_lift':
            attributions = self.deep_lift.attribute(
                input_tensor,
                target=target_class
            )
        elif method == 'gradient_shap':
            attributions = self.gradient_shap.attribute(
                input_tensor,
                n_samples=200,
                stdevs=0.0001,
                target=target_class
            )
        
        return {
            'attributions': attributions,
            'aggregated': torch.mean(torch.abs(attributions), dim=1)
        }
