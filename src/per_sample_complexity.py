
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from functools import partial

class DriftDetector:
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        
    def detect_drift(self, complexity):
        self.history.append(complexity)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        if len(self.history) < self.window_size:
            return False
            
        mean_first_half = torch.tensor(self.history[:self.window_size//2]).mean()
        mean_second_half = torch.tensor(self.history[self.window_size//2:]).mean()
        
        return abs(mean_first_half - mean_second_half) > self.threshold

def process_group(model, subset_data, complexity):
    return model(subset_data, complexity)

def redistribute_complexity_groups(grouped_data, drift_detector, resource_threshold=0.8):
    for complexity, indices in grouped_data.items():
        if drift_detector.detect_drift(complexity):
            samples_to_move = len(indices) // 2
            samples = indices[:samples_to_move]
            grouped_data[complexity] = indices[samples_to_move:]
            target_group = min(grouped_data, key=lambda k: len(grouped_data[k]))
            grouped_data[target_group].extend(samples)
    return grouped_data

def process_batch_dynamic(model, data, complexities, device):
    drift_detector = DriftDetector()
    grouped_data = defaultdict(list)
    
    for i, complexity in enumerate(complexities.values()):
        grouped_data[complexity.item()].append(i)
        
    grouped_data = redistribute_complexity_groups(grouped_data, drift_detector)
    output = torch.zeros(data.size(0), model.output_size).to(device)
    
    model_fn = partial(process_group, model)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for complexity, indices in grouped_data.items():
            subset_data = data.index_select(0, torch.tensor(indices, dtype=torch.long, device=device))
            results.append(pool.apply_async(model_fn, args=(subset_data, complexity)))
            
        for complexity, indices in grouped_data.items():
            subset_output = results.pop(0).get()
            output.index_copy_(0, torch.tensor(indices, dtype=torch.long, device=device), subset_output)
            
    return output
