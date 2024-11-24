
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_annealing(epoch, total_epochs, annealing_start_epoch):
    if epoch < annealing_start_epoch:
        return 0.0
    progress = (epoch - annealing_start_epoch) / (total_epochs - annealing_start_epoch)
    return 0.5 * (1 + math.cos(math.pi * progress))

class LearnableThresholds(nn.Module):
    def __init__(self, initial_values):
        super(LearnableThresholds, self).__init__()
        self.threshold_variance = nn.Parameter(torch.tensor(initial_values['variance']))
        self.threshold_entropy = nn.Parameter(torch.tensor(initial_values['entropy']))
        self.threshold_sparsity = nn.Parameter(torch.tensor(initial_values['sparsity']))

    def forward(self, var, ent, spar):
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

    def forward(self, var, ent, spar, current_epoch):
        annealing_weight = cosine_annealing(current_epoch, self.total_epochs, self.annealing_start_epoch)
        statistical_thresholds = self.calculate_statistical_thresholds(var, ent, spar)
        learnable_complexities = self.learnable_thresholds(var, ent, spar)
        combined_complexities = {
            key: annealing_weight * learnable_complexities[key] +
                  (1 - annealing_weight) * statistical_thresholds[key]
            for key in statistical_thresholds
        }
        return combined_complexities

    def calculate_statistical_thresholds(self, var, ent, spar):
        thresholds = {
            'variance': torch.quantile(var, 0.5),
            'entropy': torch.quantile(ent, 0.5),
            'sparsity': torch.quantile(spar, 0.5)
        }
        return thresholds
    