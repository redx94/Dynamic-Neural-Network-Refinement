import torch
from src.model import ConditionalGAN
from src.analyzer import Analyzer

def generate_synthetic_data(generator, label, num_samples=1000):
    """
    Generate synthetic data conditioned on a specific complexity label.
    """
    label_tensor = torch.nn.functional.one_hot(torch.tensor([label]*num_samples), num_classes=3).float()
    noise = torch.randn(num_samples, 100)
    synthetic_data = generator(noise, label_tensor)
    return synthetic_data

def main():
    generator = ConditionalGAN()
    generator.load_state_dict(torch.load('models/gans/generator.pth'))
    generator.to('cuda' if torch.cuda.is_available() else 'cpu')
    generator.eval()
    
    # Generate synthetic data for each complexity label
    synthetic_data_simple = generate_synthetic_data(generator, label=0, num_samples=500)
    synthetic_data_moderate = generate_synthetic_data(generator, label=1, num_samples=500)
    synthetic_data_complex = generate_synthetic_data(generator, label=2, num_samples=500)
    
    # Save synthetic data
    torch.save(synthetic_data_simple, 'data/synthetic/synthetic_simple.pth')
    torch.save(synthetic_data_moderate, 'data/synthetic/synthetic_moderate.pth')
    torch.save(synthetic_data_complex, 'data/synthetic/synthetic_complex.pth')

if __name__ == "__main__":
    main()
