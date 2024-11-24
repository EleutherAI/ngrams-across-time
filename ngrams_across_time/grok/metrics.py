
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from scipy import stats


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

