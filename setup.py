
from setuptools import setup, find_packages

setup(
    name="dynamic-neural-network",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'loguru',
        'pyyaml'
    ]
)
