from typing import Dict, Any, Callable, Optional
import optuna
from dataclasses import dataclass
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SearchSpace:
    name: str
    type: str
    bounds: tuple
    log_scale: bool = False

class HyperparameterOptimizer:
    def __init__(self,
                 search_spaces: List[SearchSpace],
                 objective_fn: Callable[[Dict[str, Any]], float],
                 n_trials: int = 100,
                 n_parallel: int = 4):
        self.search_spaces = search_spaces
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.n_parallel = n_parallel
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10)
        )
        
    def optimize(self) -> Dict[str, Any]:
        def objective(trial):
            params = {}
            for space in self.search_spaces:
                if space.type == 'float':
                    params[space.name] = trial.suggest_float(
                        space.name, 
                        space.bounds[0], 
                        space.bounds[1], 
                        log=space.log_scale
                    )
                elif space.type == 'int':
                    params[space.name] = trial.suggest_int(
                        space.name,
                        space.bounds[0],
                        space.bounds[1]
                    )
                    
            return self.objective_fn(params)
            
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_parallel
        )
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'history': self.study.trials_dataframe()
        }
