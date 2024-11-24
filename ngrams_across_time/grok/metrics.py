
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from scipy import signal, stats


def mean_l2(tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(tensor, ord=2, dim=1).mean()

def var_trace(tensor: Tensor) -> Tensor:
    return tensor.var(dim=1).sum()

def abs_score_entropy(vec: ndarray):
    abs_vec = np.abs(vec)
    sum_scores = abs_vec.sum()
    probs = abs_vec / sum_scores

    return stats.entropy(probs)

def gini(vec: ndarray):
    n = len(vec)
    diffs = np.abs(np.subtract.outer(vec, vec)).mean()

    return np.sum(diffs) / (2 * n**2 * np.mean(vec))

def hoyer(vec: ndarray):
    l1 = np.linalg.norm(vec, 1)
    l2 = np.linalg.norm(vec, 2)

    return l1 / l2

def hoyer_square(vec: ndarray):
    return hoyer(vec ** 2)


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
