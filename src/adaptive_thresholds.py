
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_sensitivity_analysis(model, data, criterion, complexities):
    gradients = []
    model.zero_grad()
    output = model(data)
    loss = criterion(output)
    complexity_loss = sum(complexities.values()).mean()
    total_loss = loss + 0.1 * complexity_loss
    total_loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info = {
                'name': name,
                'mean_grad': param.grad.abs().mean().item(),
                'std_grad': param.grad.std().item()
            }
            gradients.append(grad_info)
    return gradients

class LearnableThresholds(nn.Module):
    def __init__(self, initial_values):
        super(LearnableThresholds, self).__init__()
        self.threshold_variance = nn.Parameter(torch.tensor(initial_values['variance']))
        self.threshold_entropy = nn.Parameter(torch.tensor(initial_values['entropy']))
        self.threshold_sparsity = nn.Parameter(torch.tensor(initial_values['sparsity']))
        self.sensitivity_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, var, ent, spar, gradient_sensitivity=None):
        if gradient_sensitivity is not None:
            sensitivity_factor = torch.sigmoid(self.sensitivity_weight * gradient_sensitivity)
            return {
                'variance': self.threshold_variance * sensitivity_factor,
                'entropy': self.threshold_entropy * sensitivity_factor,
                'sparsity': self.threshold_sparsity * sensitivity_factor
            }
        return {
            'variance': self.threshold_variance,
            'entropy': self.threshold_entropy,
            'sparsity': self.threshold_sparsity
        }

class HybridThresholds(nn.Module):
    def __init__(self, initial_thresholds, annealing_start_epoch, total_epochs):
        super(HybridThresholds, self).__init__()
        self.learnable_thresholds = LearnableThresholds(initial_thresholds)
        self.annealing_start_epoch = annealing_start_epoch
        self.total_epochs = total_epochs

    def forward(self, var, ent, spar, current_epoch, gradient_sensitivity=None):
        annealing_weight = self.cosine_annealing(current_epoch)
        statistical_thresholds = self.calculate_statistical_thresholds(var, ent, spar)
        learnable_complexities = self.learnable_thresholds(var, ent, spar, gradient_sensitivity)
        
        return {
            key: annealing_weight * learnable_complexities[key] +
                 (1 - annealing_weight) * statistical_thresholds[key]
            for key in statistical_thresholds
        }

    def cosine_annealing(self, epoch):
        if epoch < self.annealing_start_epoch:
            return 0.0
        progress = (epoch - self.annealing_start_epoch) / (self.total_epochs - self.annealing_start_epoch)
        return 0.5 * (1 + math.cos(math.pi * progress))

    def calculate_statistical_thresholds(self, var, ent, spar):
        return {
            'variance': torch.quantile(var, 0.5),
            'entropy': torch.quantile(ent, 0.5),
            'sparsity': torch.quantile(spar, 0.5)
        }
