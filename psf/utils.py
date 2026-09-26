"""Numerical utilities."""
import numpy as np


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log(sum(exp(a)))."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out.ravel()[0]
    return out


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def ensure_positive(x: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    return np.maximum(x, floor)


def one_hot(idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(idx), n_classes))
    out[np.arange(len(idx)), idx] = 1.0
    return out