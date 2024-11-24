from setuptools import setup, find_packages

setup(
    name='dynamic_neural_network_refinement',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pytest',
        'wandb',
        'ray[tune]',
        'plotly',
        'dash',
        'captum'
    ],
    author='Reece Dixon',
    author_email='qtt@null.net',
    description='A dynamic neural network refinement framework with adaptive thresholds and scalability optimizations.',
    url='https://github.com/redx94/Dynamic-Neural-Network-Refinement',
    license='GNUA v3',
)
