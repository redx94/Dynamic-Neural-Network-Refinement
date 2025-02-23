# scripts/utils.py

import torch
import logging
import json_log_formatter
import os

def setup_logging(log_file):
    """Setup logging with JSON formatting."""
    formatter = json_log_formatter.JSONFormatter()
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('project_logger')
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def save_model(model, path):
    """Save the PyTorch model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load the PyTorch model."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
