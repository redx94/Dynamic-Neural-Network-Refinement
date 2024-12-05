from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
from src.models.neural_network import DynamicNeuralNetwork
from src.models.refinement import NetworkRefinement
from src.training.trainer import Trainer
from src.data.dataset import generate_synthetic_data, create_data_loaders
from src.utils.visualization import Visualizer

app = Flask(__name__)

# Initialize global objects
model = DynamicNeuralNetwork(input_size=10, output_size=1)
trainer = Trainer(model)
refinement = NetworkRefinement(model)

@app.route('/')
def index():
    """Render main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/train', methods=['POST'])
def train():
    """Handle training request"""
    # Generate synthetic data
    X, y = generate_synthetic_data()
    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:], y[800:]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        refinement_frequency=5
    )
    
    # Generate visualizations
    train_loss, val_loss = trainer.get_learning_curves()
    learning_curves = Visualizer.plot_learning_curves(train_loss, val_loss)
    
    # Get network structure
    layer_sizes = [model.input_size] + \
                 [layer.linear.out_features for layer in model.layers] + \
                 [model.output_size]
    network_structure = Visualizer.plot_network_structure(layer_sizes)
    
    # Get layer metrics
    metrics = refinement.analyze_network()
    layer_metrics = Visualizer.plot_layer_metrics(
        metrics['complexity'],
        metrics['sparsity']
    )
    
    return jsonify({
        'learning_curves': learning_curves,
        'network_structure': network_structure,
        'layer_metrics': layer_metrics,
        'final_train_loss': float(train_loss[-1]),
        'final_val_loss': float(val_loss[-1])
    })

@app.route('/refine', methods=['POST'])
def refine():
    """Handle refinement request"""
    made_changes = refinement.refine_architecture()
    
    # Get updated network structure
    layer_sizes = [model.input_size] + \
                 [layer.linear.out_features for layer in model.layers] + \
                 [model.output_size]
    network_structure = Visualizer.plot_network_structure(layer_sizes)
    
    return jsonify({
        'made_changes': made_changes,
        'network_structure': network_structure,
        'layer_count': len(model.layers)
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
