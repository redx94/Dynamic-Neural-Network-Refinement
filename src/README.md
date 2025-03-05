# Source Code Directory

The `src/` directory contains the core source code for the **Dynamic Neural Network Refinement** project. This directory is structured to promote modularity, scalability, and maintainability, adhering to best practices in software engineering and machine learning development.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Modules and Components](#modules-and-components)
  - [Model Architecture](#model-architecture)
  - [Analyzer](#analyzer)
  - [Hybrid Thresholds](#hybrid-thresholds)
  - [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
  - [Visualization](#visualization)
  - [Metrics](#metrics)
  - [Conditional GAN](#conditional-gan)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The `src/` directory houses all the essential components required to build, train, refine, and deploy dynamic neural network models. Each module is designed to perform specific functions, enabling seamless integration and efficient workflow management.

## Directory Structure

```
src/
├── __init__.py
├── ConditionalGAN.py
├── analyzer.py
├── hybrid_thresholds.py
├── layers.py
├── metrics.py
├── model.py
├── nas.py
├── visualization.py
└── app.py
```

## Modules and Components

### Model Architecture

- **`model.py`**
  Defines the `DynamicNeuralNetwork` class, which implements a neural network that dynamically adjusts its architecture based on complexity metrics such as variance, entropy, and sparsity.

  ```python
  from src.hybrid_thresholds import HybridThresholds
  from src.layers import BaseLayer
  import torch.nn as nn

  class DynamicNeuralNetwork(nn.Module):
      def __init__(self, hybrid_thresholds):
          super(DynamicNeuralNetwork, self).__init__()
          self.hybrid_thresholds = hybrid_thresholds

          # Define layers using modular BaseLayer
          self.layer1 = BaseLayer(100, 256)
          self.layer2 = BaseLayer(256, 256)
          self.layer3 = BaseLayer(256, 128)
          self.output_layer = nn.Linear(128, 10)

      def forward(self, x: torch.Tensor, complexities: dict) -> torch.Tensor:
          """
          Routes data through different layers based on complexity metrics.

          Args:
              x: Input tensor
              complexities: Dict containing variance, entropy, and sparsity metrics

          Returns:
              Output tensor after forward pass
          """
          x = self.layer1(x)
          x = self.layer2(x)

          if self._should_use_deep_path(complexities):
              x = self.layer3(x)

          return self.output_layer(x)

      def _should_use_deep_path(self, complexities: dict) -> bool:
          """Determine if deep path should be used based on complexities."""
          return (complexities['variance'].mean().item() > 0.5 and
                  complexities['entropy'].mean().item() > 0.5 and
                  complexities['sparsity'].mean().item() < 0.5)
  ```

### Analyzer

- **`analyzer.py`**
  Contains the `Analyzer` class, which computes complexity metrics of input data to inform dynamic model adjustments.

  ```python
  import torch
  import torch.nn.functional as F

  class Analyzer:
      """
      Analyzer class to compute complexity metrics of input data.
      """

      def compute_variance(self, data: torch.Tensor) -> torch.Tensor:
          var = torch.var(data, dim=1)
          return var

      def compute_entropy(self, data: torch.Tensor) -> torch.Tensor:
          probabilities = F.softmax(data, dim=1)
          entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
          return entropy

      def compute_sparsity(self, data: torch.Tensor) -> torch.Tensor:
          threshold = 0.1  # Example threshold
          sparsity = torch.mean((data.abs() < threshold).float(), dim=1)
          return sparsity

      def analyze(self, data: torch.Tensor) -> dict:
          variance = self.compute_variance(data)
          entropy = self.compute_entropy(data)
          sparsity = self.compute_sparsity(data)
          complexities = {
              'variance': variance,
              'entropy': entropy,
              'sparsity': sparsity
          }
          return complexities
  ```

### Hybrid Thresholds

- **`hybrid_thresholds.py`**
  Implements the `HybridThresholds` class, which manages dynamic threshold adjustments based on an annealing schedule to optimize model refinement.

  ```python
  import torch

  class HybridThresholds:
      """
      Handles dynamic threshold adjustments based on annealing schedule.
      """

      def __init__(self, initial_thresholds, annealing_start_epoch, total_epochs):
          self.initial_thresholds = initial_thresholds
          self.annealing_start_epoch = annealing_start_epoch
          self.total_epochs = total_epochs

      def anneal_thresholds(self, current_epoch):
          if current_epoch < self.annealing_start_epoch:
              return self.initial_thresholds
          else:
              progress = (current_epoch - self.annealing_start_epoch) / (self.total_epochs - self.annealing_start_epoch)
              annealed_thresholds = {k: v * (1 - progress) for k, v in self.initial_thresholds.items()}
              return annealed_thresholds

      def __call__(self, variance, entropy, sparsity, current_epoch):
          thresholds = self.anneal_thresholds(current_epoch)
          thresholded = {
              'variance': variance > thresholds['variance'],
              'entropy': entropy > thresholds['entropy'],
              'sparsity': sparsity < thresholds['sparsity']
          }
          return thresholded
  ```

### Neural Architecture Search (NAS)

- **`nas.py`**
  Contains the `NAS` class, which performs Neural Architecture Search by mutating and evaluating different model architectures to identify the most optimal configuration.

  ```python
  import torch
  import torch.nn as nn
  import copy
  import random

  class NAS:
      """
      A simple Neural Architecture Search (NAS) class that mutates model architectures and evaluates them.
      """
      def __init__(self, base_model, search_space, device='cpu'):
          self.base_model = base_model
          self.search_space = search_space
          self.device = device
          self.best_model = None
          self.best_score = float('-inf')

      def mutate(self, model):
          mutated_model = copy.deepcopy(model)
          mutation_type = random.choice(['add_layer', 'remove_layer', 'increase_units', 'decrease_units'])

          if mutation_type == 'add_layer' and self.search_space.get('add_layer', False):
              new_layer = nn.Linear(256, 128)
              if hasattr(mutated_model, 'layer3') and isinstance(mutated_model.layer3, nn.Sequential):
                  mutated_model.layer3 = nn.Sequential(
                      mutated_model.layer3,
                      new_layer
                  )
              else:
                  mutated_model.layer3 = nn.Sequential(new_layer)
          elif mutation_type == 'remove_layer' and self.search_space.get('remove_layer', False):
              if hasattr(mutated_model, 'layer3'):
                  delattr(mutated_model, 'layer3')
          elif mutation_type == 'increase_units' and self.search_space.get('increase_units', False):
              if hasattr(mutated_model.layer1, 'layer'):
                  mutated_model.layer1.layer = nn.Linear(mutated_model.layer1.layer.in_features, 512)
          elif mutation_type == 'decrease_units' and self.search_space.get('decrease_units', False):
              if hasattr(mutated_model.layer1, 'layer'):
                  mutated_model.layer1.layer = nn.Linear(mutated_model.layer1.layer.in_features, 128)

          return mutated_model

      def evaluate(self, model, dataloader):
          model.eval()
          correct = 0
          total = 0
          with torch.no_grad():
              for data, labels in dataloader:
                  data, labels = data.to(self.device), labels.to(self.device)
                  complexities = {
                      'variance': torch.tensor([0.6]*data.size(0)).to(self.device),
                      'entropy': torch.tensor([0.6]*data.size(0)).to(self.device),
                      'sparsity': torch.tensor([0.4]*data.size(0)).to(self.device)
                  }
                  outputs = model(data, complexities)
                  _, predicted = torch.max(outputs.data, 1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
          accuracy = correct / total
          return accuracy

      def run(self, dataloader, generations=5, population_size=5):
          population = [copy.deepcopy(self.base_model) for _ in range(population_size)]

          for gen in range(generations):
              print(f"Generation {gen+1}")
              for i in range(population_size):
                  mutated_model = self.mutate(population[i])
                  mutated_model.to(self.device)
                  score = self.evaluate(mutated_model, dataloader)
                  print(f"Model {i+1} Accuracy: {score}")
                  if score > self.best_score:
                      self.best_score = score
                      self.best_model = mutated_model
          print(f"Best Accuracy: {self.best_score}")
          return self.best_model
  ```

### Visualization

- **`visualization.py`**
  Implements functions to visualize training metrics and model performance, aiding in monitoring and analysis.

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd
  import os

  def plot_training_metrics(metrics, epoch, output_dir='visualizations/training_plots/'):
      os.makedirs(output_dir, exist_ok=True)

      # Plot Loss
      plt.figure(figsize=(10, 5))
      sns.lineplot(x=metrics['epoch'], y=metrics['loss'])
      plt.title('Training Loss over Epochs')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.savefig(os.path.join(output_dir, f'training_loss_epoch_{epoch}.png'))
      plt.close()

      # Plot Accuracy
      plt.figure(figsize=(10, 5))
      sns.lineplot(x=metrics['epoch'], y=metrics['accuracy'])
      plt.title('Training Accuracy over Epochs')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.savefig(os.path.join(output_dir, f'training_accuracy_epoch_{epoch}.png'))
      plt.close()
  ```

### Metrics

- **`metrics.py`**
  Defines Prometheus metrics to monitor the application's performance and health.

  ```python
  from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
  from fastapi import FastAPI, Response
  import time

  # Define metrics
  REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests')
  REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Latency of requests in seconds')

  def setup_metrics(app: FastAPI):
      @app.middleware("http")
      async def prometheus_middleware(request, call_next):
          REQUEST_COUNT.inc()
          start_time = time.time()
          response = await call_next(request)
          REQUEST_LATENCY.observe(time.time() - start_time)
          return response

      @app.get("/metrics")
      async def metrics():
          data = generate_latest()
          return Response(content=data, media_type=CONTENT_TYPE_LATEST)
  ```

### Conditional GAN

- **`ConditionalGAN.py`**
  Implements a Conditional Generative Adversarial Network (GAN) for generating synthetic data conditioned on specific inputs or labels.

  ```python
  import torch
  import torch.nn as nn

  class Generator(nn.Module):
      def __init__(self, input_dim, condition_dim, output_dim):
          super(Generator, self).__init__()
          self.model = nn.Sequential(
              nn.Linear(input_dim + condition_dim, 256),
              nn.ReLU(),
              nn.Linear(256, 512),
              nn.ReLU(),
              nn.Linear(512, output_dim),
              nn.Tanh()
          )

      def forward(self, noise, conditions):
          x = torch.cat((noise, conditions), dim=1)
          return self.model(x)

  class Discriminator(nn.Module):
      def __init__(self, input_dim, condition_dim):
          super(Discriminator, self).__init__()
          self.model = nn.Sequential(
              nn.Linear(input_dim + condition_dim, 512),
              nn.LeakyReLU(0.2),
              nn.Linear(512, 256),
              nn.LeakyReLU(0.2),
              nn.Linear(256, 1),
              nn.Sigmoid()
          )

      def forward(self, data, conditions):
          x = torch.cat((data, conditions), dim=1)
          return self.model(x)
  ```

## Usage

To utilize the modules and components within the `src/` directory, follow these general guidelines:

1. **Importing Modules:**

   ```python
   from src.model import DynamicNeuralNetwork
   from src.analyzer import Analyzer
   from src.hybrid_thresholds import HybridThresholds
   from src.nas import NAS
   from src.visualization import plot_training_metrics
   from src.metrics import setup_metrics
   from src.ConditionalGAN import Generator, Discriminator
   ```

2. **Initializing Components:**

   ```python
   # Initialize HybridThresholds
   initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
   hybrid_thresholds = HybridThresholds(
       initial_thresholds=initial_thresholds,
       annealing_start_epoch=5,
       total_epochs=20
   )

   # Initialize Model
   model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds)

   # Initialize Analyzer
   analyzer = Analyzer()

   # Initialize NAS
   search_space = {
       'add_layer': True,
       'remove_layer': True,
       'increase_units': True,
       'decrease_units': True
   }
   nas = NAS(
       base_model=model,
       search_space=search_space,
       device='cuda' if torch.cuda.is_available() else 'cpu'
   )
   ```

3. **Integrating with FastAPI:**

   The `app.py` file in the `src/` directory sets up the FastAPI application, integrating the model for serving predictions and exposing metrics.

   ```python
   from fastapi import FastAPI
   from src.metrics import setup_metrics
   from src.model import DynamicNeuralNetwork
   from src.hybrid_thresholds import HybridThresholds
   from src.analyzer import Analyzer

   app = FastAPI(title="Dynamic Neural Network Refinement API")

   # Setup metrics
   setup_metrics(app)

   # Initialize components
   hybrid_thresholds = HybridThresholds(...)
   model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds)
   analyzer = Analyzer()

   @app.post("/predict")
   def predict(...):
       ...
   ```

## Contributing

We welcome contributions to enhance the **Dynamic Neural Network Refinement** project! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "feat: add your feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

   Submit a pull request detailing your changes and their benefits.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
