
from setuptools import setup, find_packages

setup(
    name="dynamic-neural-network",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.5',
        'pandas>=1.3.0',
        'matplotlib>=3.4.3',
        'seaborn>=0.11.2',
        'loguru>=0.7.0',
        'pyyaml',
        'scikit-learn>=1.0.2',
        'wandb>=0.12.0',
        'tensorboard>=2.6.0'
    ]
)
