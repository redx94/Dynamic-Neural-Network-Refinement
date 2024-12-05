# scripts/train.py

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from src.visualization import plot_training_metrics
from scripts.utils import save_model, setup_logging
import wandb
import yaml
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dynamic Neural Network")
    parser.add_argument('--config', type=str, help='Path to training configuration file', required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_distributed(config):
    if config['distributed']['enabled']:
        dist.init_process_group(
            backend=config['distributed']['backend'],
            init_method=config['distributed']['init_method']
        )
        torch.cuda.set_device(config['distributed']['local_rank'])
        device = torch.device('cuda', config['distributed']['local_rank'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return rank, world_size, device

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup distributed training
    rank, world_size, device = setup_distributed(config)
    
    # Initialize logging (only on rank 0)
    if rank == 0:
        logger = setup_logging('logs/train.json.log')
        logger.info("Starting training process...")
    else:
        logger = None
    
    # Initialize W&B only on rank 0
    if rank == 0:
        wandb.init(project=config['logging']['wandb']['project'], entity=config['logging']['wandb']['entity'], config=config)
    
    # Initialize components
    analyzer = Analyzer()
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = getattr(torch.nn, config['training']['loss_function'])()
    
    # Placeholder for storing metrics
    metrics_history = {
        'epoch': [],
        'loss': [],
        'accuracy': []
    }
    
    # Example training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        # Replace the following with actual distributed data loading
        dummy_input = torch.randn(config['training']['batch_size'], config['model']['input_size']).to(device)
        dummy_labels = torch.randint(0, config['model']['output_size'], (config['training']['batch_size'],)).to(device)
        
        complexities = analyzer.analyze(dummy_input)
        complexities = hybrid_thresholds(complexities['variance'], complexities['entropy'], complexities['sparsity'], current_epoch=epoch)
        outputs = model(dummy_input, complexities)
        loss = loss_fn(outputs, dummy_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy (dummy)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == dummy_labels).float().mean().item()
        
        # Log metrics
        if rank == 0:
            wandb.log({'epoch': epoch, 'loss': loss.item(), 'accuracy': accuracy})
            logger.info(f"Epoch {epoch}: Loss={loss.item()}, Accuracy={accuracy}")
        
        # Store metrics
        metrics_history['epoch'].append(epoch)
        metrics_history['loss'].append(loss.item())
        metrics_history['accuracy'].append(accuracy)
        
        # Save checkpoints
        if rank == 0 and epoch % config['training']['checkpoints']['save_interval'] == 0:
            checkpoint_path = f"{config['training']['checkpoints']['checkpoint_dir']}/model_epoch_{epoch}.pth"
            save_model(model.module if world_size > 1 else model, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
    
    # Save final model
    if rank == 0:
        final_model_path = config['output']['final_model_path']
        save_model(model.module if world_size > 1 else model, final_model_path)
        logger.info(f"Final model saved at {final_model_path}")
        
        # Generate training plots
        plot_training_metrics(metrics_history, epoch=config['training']['epochs'])
        
        wandb.finish()
        logger.info("Training process completed.")
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
