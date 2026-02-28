import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import torch
from scipy import stats

@dataclass
class ExperimentVariant:
    name: str
    config: Dict[str, Any]
    weight: float = 1.0

class ABTestingManager:
    def __init__(self,
                 experiment_name: str,
                 variants: List[ExperimentVariant],
                 success_metric: Callable[[Dict[str, float]], float],
                 significance_level: float = 0.05):
        self.experiment_name = experiment_name
        self.variants = variants
        self.success_metric = success_metric
        self.significance_level = significance_level
        self.results_per_variant: Dict[str, List[float]] = {
            v.name: [] for v in variants
        }
        
    def assign_variant(self) -> ExperimentVariant:
        weights = np.array([v.weight for v in self.variants])
        weights = weights / weights.sum()
        return np.random.choice(self.variants, p=weights)
    
    def record_result(self, variant_name: str, metrics: Dict[str, float]):
        success_value = self.success_metric(metrics)
        self.results_per_variant[variant_name].append(success_value)
        
    def analyze_results(self) -> Dict[str, Any]:
        if len(self.variants) < 2:
            return {'error': 'Need at least 2 variants for analysis'}
            
        stats_results = {}
        control = self.variants[0].name
        
        for variant in self.variants[1:]:
            t_stat, p_value = stats.ttest_ind(
                self.results_per_variant[control],
                self.results_per_variant[variant.name]
            )
            
            stats_results[variant.name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'mean_difference': (
                    np.mean(self.results_per_variant[variant.name]) -
                    np.mean(self.results_per_variant[control])
                )
            }
            
        return stats_results
