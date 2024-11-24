
import numpy as np
from torch import Tensor
import torch
from scipy import signal


def mean_l2(tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(tensor, ord=2, dim=1).mean()


def var_trace(tensor: Tensor) -> Tensor:
    return tensor.var(dim=1).sum()


def gini(vec):
    n: int = len(vec)
    diffs = np.abs(np.subtract.outer(vec, vec)).mean()
    return np.sum(diffs) / (2 * n**2 * np.mean(vec))


def hoyer(vec):
    return np.linalg.norm(vec, 1) / np.linalg.norm(vec, 2)


def hoyer_square(vec):
    vec_squared = np.array(vec) ** 2
    return np.linalg.norm(vec_squared, 1) / np.linalg.norm(vec_squared, 2)


# Unused
def spectral_entropy(signal_data, sf, nperseg=None, normalize=True):
    """
    Calculate spectral entropy of a time series signal.
    
    Parameters:
    -----------
    signal_data : array-like
        Input signal - time series data
    sf : float
        Sampling frequency of the signal
    nperseg : int, optional
        Length of each segment for Welch's method
        If None, defaults to sf*2 (2 second windows)
    normalize : bool, optional
        If True, entropy is normalized by log2(n_freq_bins)
    
    Returns:
    --------
    float
        Spectral entropy value
    dict
        Additional information including PSD and frequencies
    """
    # Input validation
    signal_data = np.array(signal_data)
    if signal_data.size == 0:
        raise ValueError("Input signal is empty")
    
    # Set default nperseg if not specified
    if nperseg is None:
        nperseg = int(sf * 2)
    
    _, psd = signal.welch(signal_data, sf, nperseg=nperseg)
    
    psd_norm = psd / psd.sum()
    
    spec_ent = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))
    
    if normalize:
        spec_ent /= np.log2(len(psd_norm))
    
    return spec_ent
