import torch
import time
from typing import Dict, List, Callable
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class BenchmarkMetrics:
    inference_time: float
    memory_usage: float
    flops: int
    parameter_count: int
    adaptation_overhead: float

class PerformanceBenchmark:
    def __init__(self, 
                 model: nn.Module,
                 input_shapes: List[tuple],
                 device: str = 'cuda'):
        self.model = model
        self.input_shapes = input_shapes
        self.device = device
        self.metrics_history = []
        
    def run_comprehensive_benchmark(self) -> Dict[str, float]:
        metrics = {}
        
        # Measure inference time
        metrics['inference_time'] = self._benchmark_inference()
        
        # Measure memory usage
        metrics['memory'] = self._measure_memory_usage()
        
        # Count FLOPs
        metrics['flops'] = self._count_flops()
        
        # Measure adaptation overhead
        metrics['adaptation_time'] = self._measure_adaptation_overhead()
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _benchmark_inference(self, num_runs: int = 100) -> float:
        times = []
        for shape in self.input_shapes:
            input_tensor = torch.randn(shape).to(self.device)
            
            # Warmup
            for _ in range(10):
                self.model(input_tensor)
                
            # Actual measurement
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            for _ in range(num_runs):
                self.model(input_tensor)
                
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) / num_runs)
            
        return np.mean(times)
