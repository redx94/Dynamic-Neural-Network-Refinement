import torch
import torch.autograd.profiler as profiler
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ProfileMetrics:
    cpu_time: float
    cuda_time: float
    memory_usage: int
    flops: int
    bottlenecks: List[str]

class AutoProfiler:
    def __init__(self, 
                 enabled: bool = True,
                 trace_memory: bool = True,
                 trace_cuda: bool = True):
        self.enabled = enabled
        self.trace_memory = trace_memory
        self.trace_cuda = trace_cuda
        self.history = []
        
    @contextmanager
    def profile_section(self, section_name: str):
        if not self.enabled:
            yield
            return
            
        with profiler.profile(
            use_cuda=self.trace_cuda,
            profile_memory=self.trace_memory
        ) as prof:
            yield
            
        metrics = self._analyze_profile(prof, section_name)
        self.history.append(metrics)
        
    def _analyze_profile(self, 
                        prof: profiler.ProfilerResult,
                        section_name: str) -> ProfileMetrics:
        events_df = pd.DataFrame(prof.key_averages())
        
        # Calculate key metrics
        cpu_time = events_df['cpu_time_total'].sum()
        cuda_time = events_df['cuda_time_total'].sum() if self.trace_cuda else 0
        memory = events_df['cpu_memory_usage'].sum()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(events_df)
        
        return ProfileMetrics(
            cpu_time=cpu_time,
            cuda_time=cuda_time,
            memory_usage=memory,
            flops=self._estimate_flops(events_df),
            bottlenecks=bottlenecks
        )
