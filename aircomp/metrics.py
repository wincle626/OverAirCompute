"""Error metrics for over-the-air computation."""

from __future__ import annotations

import numpy as np

from .utils import linear_to_db


def mse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean-squared error between estimated and true aggregates."""
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    return float(np.mean(np.abs(estimate - truth) ** 2))


def nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Normalised MSE = ``E|est - truth|^2 / E|truth|^2`` (linear)."""
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    power = np.mean(np.abs(truth) ** 2)
    if power < 1e-30:
        return float("nan")
    return float(np.mean(np.abs(estimate - truth) ** 2) / power)


def nmse_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Normalised MSE in dB."""
    return float(linear_to_db(nmse(estimate, truth)))
