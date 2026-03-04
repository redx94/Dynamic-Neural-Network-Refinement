import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDistiller:
    """
    Continuous Edge-Device Knowledge Distillation Suite.
    
    This module takes the high-accuracy predictions from the heavy Cloud Engine (Teacher)
    and uses them to generate soft targets to train the local Edge Engine (Student)
    in the background.
    
    This allows local IoT and Edge Devices to get smarter every day without sending their
    raw data back to the cloud for massive retraining cycles.
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            temperature: Softens the teacher's probability distribution. Higher T = softer targets.
            alpha: Balances the loss between hard labels (student) and soft labels (teacher).
        """
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss_func = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_func = nn.CrossEntropyLoss()

    def compute_distillation_loss(self, student_logits: torch.Tensor, 
                                  teacher_logits: torch.Tensor, 
                                  targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates how far the Student (Shallow Path) is from the Teacher (Deep Path).
        Returns a blended loss used to gradient-update the Student model.
        """
        # 1. Soften both the Student and Teacher outputs using Temperature
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_prob = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 2. Calculate Distillation Loss (KL Divergence between Student and Teacher)
        # Multiply by T^2 to balance the magnitude against standard CE Loss
        distillation_loss = self.kl_loss_func(soft_targets, soft_prob) * (self.temperature ** 2)

        # 3. Calculate Standard Classification Loss (Cross Entropy on hard labels)
        student_loss = self.ce_loss_func(student_logits, targets)

        # 4. Blend the losses together using Alpha
        total_loss = (self.alpha * distillation_loss) + ((1.0 - self.alpha) * student_loss)
        
        return total_loss
