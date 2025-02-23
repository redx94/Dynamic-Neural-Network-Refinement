import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class AdaptiveKnowledgeDistillation:
    def __init__(self, 
                 temperature: float = 3.0,
                 alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_cache = {}
        
    def distill_knowledge(self, 
                         student: nn.Module,
                         teacher: nn.Module,
                         inputs: torch.Tensor,
                         targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get soft targets from teacher
        with torch.no_grad():
            teacher_logits = teacher(inputs)
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            
        # Get student predictions
        student_logits = student(inputs)
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        
        # Calculate distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        if targets is not None:
            # Combine with hard targets if available
            student_loss = F.cross_entropy(student_logits, targets)
            total_loss = (self.alpha * student_loss + 
                         (1 - self.alpha) * distillation_loss)
        else:
            total_loss = distillation_loss
            
        return total_loss
    
    def update_teacher_cache(self, 
                           layer_id: str,
                           features: torch.Tensor):
        # Cache teacher's intermediate features for later analysis
        self.teacher_cache[layer_id] = features.detach()
    
    def get_layer_alignment(self, 
                          student_features: torch.Tensor,
                          layer_id: str) -> float:
        if layer_id not in self.teacher_cache:
            return 0.0
            
        teacher_features = self.teacher_cache[layer_id]
        
        # Calculate CKA (Centered Kernel Alignment) between feature spaces
        similarity = self._compute_cka(student_features, teacher_features)
        return similarity
    
    def _compute_cka(self, 
                    X: torch.Tensor,
                    Y: torch.Tensor) -> float:
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        
        hsic_xy = torch.trace(X @ Y.T @ Y @ X.T)
        hsic_xx = torch.trace(X @ X.T @ X @ X.T)
        hsic_yy = torch.trace(Y @ Y.T @ Y @ Y.T)
        
        return hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))
