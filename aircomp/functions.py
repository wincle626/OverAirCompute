"""Arithmetic operations that AirComp can compute.

Every operation is expressed through the AirComp primitive — the wireless
channel forms a **sum** of what the agents transmit — as

    f(s_1, ..., s_K) = psi( sum_k phi_k(s_k) ),      (nomographic form)

where each agent applies pre-processing ``phi_k`` (``pre``) and the host applies
post-processing ``psi`` (``post``) to the received sum.  Some operations need
several parallel sums (e.g. a histogram), so ``pre`` may return one column per
output dimension.

The operations are grouped into three families:

* **Natural-medium** — realised by the summation property directly, with no
  post-processing: addition, subtraction (signed addition).
* **Nomographic (pre-/post-processed)** — exact given a perfect sum: arithmetic
  mean, weighted average, geometric mean, Euclidean norm, polynomial sums.
* **Advanced / approximated** — realised only approximately, either because the
  result is discrete and recovered by rounding a noisy sum (majority vote,
  counting, histogram) or because the function is approximated by a p-norm
  surrogate (maximum, minimum).

Each operation also exposes ``sample_data`` (a representative data model) and
``true_value`` (the exact mathematical result) so it can be benchmarked; see
:mod:`aircomp.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardization:
    """Affine map applied to symbols so that ``x = (g - shift) / scale``.

    Transmitting standardised (unit-power) symbols and inverting the map on the
    received sum is the standard way to meet the AirComp power model; the shift
    and scale are treated as known side information at the host.
    """

    shift: float
    scale: float

    def forward(self, g: np.ndarray) -> np.ndarray:
        return (g - self.shift) / self.scale

    def invert_sum(self, x_sum: float | complex, n_terms: int) -> float:
        """Recover ``sum_k g_k`` from an estimate of ``sum_k x_k``."""
        return float(np.real(x_sum) * self.scale + n_terms * self.shift)


def standardize(values: np.ndarray) -> Standardization:
    """Build a unit-power standardisation from a vector of values."""
    values = np.asarray(values, dtype=float)
    shift = float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1e-12:
        scale = 1.0
    return Standardization(shift=shift, scale=scale)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Operation:
    """Base class for an AirComp-computable operation."""

    name = "operation"
    category = "generic"
    #: True if the result is only approximately recoverable over the air.
    approximate = False
    #: True if the exact result is integer-valued (recovered by rounding).
    discrete = False

    # -- data model -------------------------------------------------------- #
    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        """A representative dataset (default: standard normal, one per agent)."""
        return rng.standard_normal(n_agents)

    # -- nomographic decomposition ---------------------------------------- #
    def pre(self, data: np.ndarray) -> np.ndarray:
        """Per-agent pre-processing ``phi_k``.

        Returns an array of shape ``(n_agents,)`` for a scalar output, or
        ``(n_agents, D)`` when the operation needs ``D`` parallel sums.
        """
        return np.asarray(data, dtype=float)

    def post(self, aggregate_sum, n_agents: int):
        """Post-processing ``psi`` applied to the (length-``D``) aggregate sum."""
        raise NotImplementedError

    # -- benchmarking ------------------------------------------------------ #
    def true_value(self, data: np.ndarray):
        """Exact mathematical result (default: assumes an exact nomographic op)."""
        pre = np.atleast_2d(self._pre_2d(data))
        return self.post(pre.sum(axis=0), len(np.asarray(data)))

    def _pre_2d(self, data: np.ndarray) -> np.ndarray:
        pre = np.asarray(self.pre(data), dtype=float)
        return pre[:, None] if pre.ndim == 1 else pre

    @staticmethod
    def _s0(aggregate_sum) -> float:
        return float(np.ravel(aggregate_sum)[0])


# --------------------------------------------------------------------------- #
# Natural-medium operations
# --------------------------------------------------------------------------- #
class Summation(Operation):
    """Addition: ``sum_k s_k`` (the raw superposition result)."""

    name = "addition"
    category = "natural-medium"

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum)

    def true_value(self, data: np.ndarray) -> float:
        return float(np.sum(data))


class Subtraction(Operation):
    """Subtraction as signed addition: ``sum_k eps_k s_k`` with ``eps_k = +/-1``.

    By default the first agent's value is added and all others subtracted,
    illustrating that subtraction is just addition of negated pre-processed
    values over the same summing channel.
    """

    name = "subtraction"
    category = "natural-medium"

    def __init__(self, signs: np.ndarray | None = None):
        self.signs = None if signs is None else np.asarray(signs, dtype=float)

    def _get_signs(self, n: int) -> np.ndarray:
        if self.signs is not None:
            return self.signs
        s = -np.ones(n)
        s[0] = 1.0
        return s

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return self._get_signs(len(data)) * data

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum)

    def true_value(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=float)
        return float(self._get_signs(len(data)) @ data)


# --------------------------------------------------------------------------- #
# Nomographic (pre-/post-processed) operations
# --------------------------------------------------------------------------- #
class ArithmeticMean(Operation):
    """Arithmetic mean ``(1/K) sum_k s_k`` — the canonical FL aggregation."""

    name = "arithmetic_mean"
    category = "nomographic"

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum) / n_agents

    def true_value(self, data: np.ndarray) -> float:
        return float(np.mean(data))


class WeightedSum(Operation):
    """Weighted sum ``sum_k w_k s_k`` (weights fold into pre-processing)."""

    name = "weighted_sum"
    category = "nomographic"

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def _get_weights(self, n: int) -> np.ndarray:
        return self.weights if self.weights is not None else np.linspace(0.5, 1.5, n)

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return self._get_weights(len(data)) * data

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum)


class WeightedAverage(Operation):
    """Weighted average ``(sum_k w_k s_k) / (sum_k w_k)``."""

    name = "weighted_average"
    category = "nomographic"

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def _get_weights(self, n: int) -> np.ndarray:
        return self.weights if self.weights is not None else np.linspace(0.5, 1.5, n)

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return self._get_weights(len(data)) * data

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum) / float(np.sum(self._get_weights(n_agents)))


class GeometricMean(Operation):
    """Geometric mean ``(prod_k s_k)^(1/K)`` for positive data.

    Realised as ``exp( (1/K) sum_k log s_k )``.
    """

    name = "geometric_mean"
    category = "nomographic"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.5, 2.0, size=n_agents)

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if np.any(data <= 0):
            raise ValueError("GeometricMean requires strictly positive data.")
        return np.log(data)

    def post(self, aggregate_sum, n_agents: int) -> float:
        return float(np.exp(self._s0(aggregate_sum) / n_agents))


class EuclideanNorm(Operation):
    """Euclidean norm ``sqrt(sum_k s_k^2)``."""

    name = "euclidean_norm"
    category = "nomographic"

    def pre(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float) ** 2

    def post(self, aggregate_sum, n_agents: int) -> float:
        return float(np.sqrt(max(self._s0(aggregate_sum), 0.0)))


class PolynomialFunction(Operation):
    """Sum of a per-agent polynomial ``sum_k p(s_k)``.

    ``coeffs`` are in ascending order, i.e. ``p(s) = sum_j coeffs[j] s^j``.
    """

    name = "polynomial_sum"
    category = "nomographic"

    def __init__(self, coeffs: np.ndarray | None = None):
        # Default: p(s) = 1 + 2 s + 3 s^2.
        self.coeffs = np.asarray(
            [1.0, 2.0, 3.0] if coeffs is None else coeffs, dtype=float
        )

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        powers = np.vstack([data ** j for j in range(len(self.coeffs))])  # (J, K)
        return self.coeffs @ powers  # (K,)

    def post(self, aggregate_sum, n_agents: int) -> float:
        return self._s0(aggregate_sum)


# --------------------------------------------------------------------------- #
# Advanced / approximated operations
# --------------------------------------------------------------------------- #
class MajorityVote(Operation):
    """Majority vote over binary inputs, via the sign of a +/-1 sum."""

    name = "majority_vote"
    category = "advanced"
    approximate = True
    discrete = True

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=n_agents).astype(float)  # {0, 1}

    def pre(self, data: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(data, dtype=float) - 1.0  # {0,1} -> {-1,+1}

    def post(self, aggregate_sum, n_agents: int) -> float:
        return 1.0 if self._s0(aggregate_sum) > 0.0 else 0.0

    def true_value(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=float)
        return 1.0 if data.sum() > len(data) / 2.0 else 0.0


class Counting(Operation):
    """Count agents whose value exceeds a threshold, via a rounded sum."""

    name = "counting"
    category = "advanced"
    approximate = True
    discrete = True

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def pre(self, data: np.ndarray) -> np.ndarray:
        return (np.asarray(data, dtype=float) > self.threshold).astype(float)

    def post(self, aggregate_sum, n_agents: int) -> float:
        return float(np.round(self._s0(aggregate_sum)))

    def true_value(self, data: np.ndarray) -> float:
        return float(np.sum(np.asarray(data, dtype=float) > self.threshold))


class Histogram(Operation):
    """Histogram counts over fixed bin edges (one parallel sum per bin)."""

    name = "histogram"
    category = "advanced"
    approximate = True
    discrete = True

    def __init__(self, edges: np.ndarray | None = None):
        # Interior edges; produces len(edges)+1 bins via np.digitize.
        self.edges = np.asarray([-1.0, 0.0, 1.0] if edges is None else edges, dtype=float)

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    def _bin_index(self, data: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(data, dtype=float), self.edges)

    def pre(self, data: np.ndarray) -> np.ndarray:
        idx = self._bin_index(data)
        onehot = np.zeros((len(idx), self.n_bins), dtype=float)
        onehot[np.arange(len(idx)), idx] = 1.0
        return onehot  # (K, n_bins)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.round(np.ravel(aggregate_sum))

    def true_value(self, data: np.ndarray) -> np.ndarray:
        idx = self._bin_index(data)
        return np.bincount(idx, minlength=self.n_bins).astype(float)


class MaxApprox(Operation):
    """Maximum approximated by the p-norm ``(sum_k s_k^p)^(1/p)`` (positive data)."""

    name = "maximum"
    category = "advanced"
    approximate = True

    def __init__(self, p: float = 6.0):
        self.p = float(p)

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.3, 1.0, size=n_agents)

    def pre(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float) ** self.p

    def post(self, aggregate_sum, n_agents: int) -> float:
        return float(max(self._s0(aggregate_sum), 0.0) ** (1.0 / self.p))

    def true_value(self, data: np.ndarray) -> float:
        return float(np.max(data))


class MinApprox(Operation):
    """Minimum approximated by ``(sum_k s_k^{-p})^{-1/p}`` (positive data)."""

    name = "minimum"
    category = "advanced"
    approximate = True

    def __init__(self, p: float = 6.0):
        self.p = float(p)

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.3, 1.0, size=n_agents)

    def pre(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float) ** (-self.p)

    def post(self, aggregate_sum, n_agents: int) -> float:
        return float(max(self._s0(aggregate_sum), 1e-30) ** (-1.0 / self.p))

    def true_value(self, data: np.ndarray) -> float:
        return float(np.min(data))


# Backwards-compatible alias: the base class was previously named this.
NomographicFunction = Operation


#: A representative operation from every family, for evaluation.
DEFAULT_OPERATIONS = [
    Summation(),
    Subtraction(),
    ArithmeticMean(),
    WeightedAverage(),
    GeometricMean(),
    EuclideanNorm(),
    PolynomialFunction(),
    MajorityVote(),
    Counting(),
    Histogram(),
    MaxApprox(),
    MinApprox(),
]
