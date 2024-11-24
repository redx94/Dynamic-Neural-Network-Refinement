
import torch

def normalize_data(data):
    return (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-6)

def preprocess_data(data, complexities):
    normalized_data = normalize_data(data)
    return normalized_data, complexities
    