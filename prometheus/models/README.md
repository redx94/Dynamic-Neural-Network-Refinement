# Models Directory

The `models/` directory stores all trained, pruned, quantized, and final versions of the neural network models used in the **Dynamic Neural Network Refinement** project. This structure ensures organized management of different model states, facilitating easy access, deployment, and version control.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Subdirectories](#subdirectories)
  - [pruned/](#pruned)
  - [quantized/](#quantized)
  - [final/](#final)
- [Usage](#usage)
  - [Accessing Models](#accessing-models)
  - [Loading Models in Scripts](#loading-models-in-scripts)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Directory Structure

```
models/
├── pruned/
├── quantized/
└── final/
```

## Subdirectories

### pruned/

- **Purpose:**  
  Stores models that have undergone pruning to remove less significant weights. Pruning reduces the model size and can enhance performance by eliminating redundancies.

- **Contents:**  
  - `pruned_model_v1.0.pth`: Example pruned model file.
  - Additional pruned models with versioning as needed.

### quantized/

- **Purpose:**  
  Contains models that have been quantized to lower precision (e.g., `int8`). Quantization further optimizes models for faster inference and reduced memory usage, especially beneficial for deployment on resource-constrained devices.

- **Contents:**  
  - `quantized_model_v1.0.pth`: Example quantized model file.
  - Additional quantized models with versioning as needed.

### final/

- **Purpose:**  
  Holds the final versions of the models ready for deployment. These models have typically undergone training, evaluation, pruning, and quantization to ensure optimal performance and efficiency.

- **Contents:**  
  - `model_v1.0_final.pth`: Example final model file.
  - `model_v1.0_final.onnx`: ONNX-exported version of the final model for broader compatibility.
  - Additional final models and their exported versions as needed.

## Usage

### Accessing Models

To utilize the models stored in the `models/` directory, reference the appropriate subdirectory based on the model's state:

- **Trained Model:**  
  Located in `final/` or `pruned/` before quantization.
  
- **Pruned Model:**  
  Located in `pruned/`.
  
- **Quantized Model:**  
  Located in `quantized/`.

### Loading Models in Scripts

When loading models within your scripts, specify the path to the desired model version. Here's an example of how to load the quantized model for inference:

```python
import torch
from src.model import DynamicNeuralNetwork

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize HybridThresholds
from src.hybrid_thresholds import HybridThresholds
initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
hybrid_thresholds = HybridThresholds(
    initial_thresholds=initial_thresholds,
    annealing_start_epoch=5,
    total_epochs=20
)

# Initialize Model
model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)

# Load Quantized Model
model_path = 'models/quantized/quantized_model_v1.0.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Now, model is ready for inference
```

## Best Practices

- **Versioning:**  
  Maintain clear and consistent versioning for all model files (e.g., `v1.0`, `v1.1`) to track changes and improvements over time.

- **Backup:**  
  Regularly back up models, especially final versions, to prevent data loss. Consider using cloud storage solutions or version control systems like Git LFS.

- **Documentation:**  
  Keep detailed documentation of each model's training parameters, pruning and quantization settings, and performance metrics to facilitate reproducibility and analysis.

- **Naming Conventions:**  
  Use descriptive and consistent naming conventions for model files to easily identify their state and version (e.g., `pruned_model_v1.0.pth`, `quantized_model_v1.0.pth`).

## Contributing

Contributions to the models directory are welcome! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/add-new-model-version
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "feat: add pruned_model_v1.1.pth"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/add-new-model-version
   ```

5. **Open a Pull Request**

   Provide a clear description of the model added, its state (pruned, quantized, etc.), and any relevant performance metrics or changes.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
