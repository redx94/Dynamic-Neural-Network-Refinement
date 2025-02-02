import numpy as np


def normalize_data(data):
    """
    Normalizes the input data using mean and standard deviation.

    Args:
        data (np.ndarray): Input dataset.

    Returns:
        np.ndarray: Normalized dataset.
    """
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)


def preprocess_data(data, complexities):
    """
    Preprocesses the input data by normalizing it and handling complexities.

    Args:
        data (np.ndarray): Input dataset.
        complexities (dict): Dictionary containing variance, entropy, and sparsity.

    Returns:
        Tuple[np.ndarray, dict]: Preprocessed data and corresponding complexities.
    """
    normalized_data = normalize_data(data)
    return normalized_data, complexities