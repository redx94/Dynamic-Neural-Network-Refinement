import argparse
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
from src.hybrid_thresholds import HybridThresholds
from scripts.utils import load_model, save_model, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dynamic Neural Network")
    parser.add_argument('--config', type=str, help='Path to training config file', required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_thresholds = HybridThresholds(
        initial_thresholds=config['thresholds'],
        annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
        total_epochs=config['thresholds']['total_epochs']
    )
    model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)

    # Load model
    model = load_model(model, args.config, device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.functional.mse_loss

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        input_data = torch.randn(
            config['training']['batch_size'], config['model']['input_size']
        ).to(device)
        complexities = {
            'variance': torch.tensor([0.6] * input_data.size(0)).to(device),
            'entropy': torch.tensor([0.6] * input_data.size(0)).to(device),
            'sparsity': torch.tensor([0.4] * input_data.size(0)).to(device)
        }
        outputs = model(input_data, complexities)
        target = torch.randn(config['training']['batch_size'], config['model']['output_size']).to(device)
        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{config['training']['epochs']} - Loss: {loss.item():.4f}")

    # Save final model
    save_model(model, config['output']['final_model_path'])
    print(f"Final model saved at {config['output']['final_model_path']}")


if __name__ == "__main__":
    main()