
import torch
from src.models.dynamic_nn import DynamicNeuralNetwork
from src.utils.data_loader import get_data_loaders
from src.utils.logging import setup_wandb

# Initialize W&B
setup_wandb(project_name="dynamic_neural_network_refinement")

# Load data
train_loader, val_loader = get_data_loaders()

# Initialize model
model = DynamicNeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):  # Replace 10 with total_epochs from configuration
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
