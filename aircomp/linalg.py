"""Vector- and matrix-valued linear-algebra operations for AirComp.

Most distributed linear algebra reduces to a **sum over agents** of a per-agent
tensor, which is exactly what the AirComp channel computes.  Each operation
flattens its per-agent contribution into a real vector of length ``D`` and is
aggregated with ``D`` channel uses (sharing one channel realisation and
beamformer); the host reshapes the aggregate and applies any post-processing.

Operations
----------
* :class:`VectorSum`, :class:`VectorMean` — element-wise vector aggregation.
* :class:`FederatedAveraging` — weighted model/update averaging ``Σ wₖ θₖ / Σ wₖ``
  (the canonical federated-learning step).
* :class:`GramMatrix` — second-moment matrix ``Σ_k xₖ xₖᵀ``.
* :class:`CovarianceMatrix` — biased sample covariance ``(1/K) Σ (xₖ−μ)(xₖ−μ)ᵀ``.
* :class:`MatrixSum` — element-wise matrix aggregation ``Σ_k Mₖ``.
* :class:`DistributedMatrixVector` — product ``(Σ_k Aₖ) x`` of a distributed
  matrix with a shared vector, via ``Σ_k (Aₖ x)``.
* :class:`DistributedLeastSquares` — ridge/least-squares weights from the
  over-the-air normal equations ``XᵀX = Σ xₖxₖᵀ`` and ``Xᵀy = Σ xₖ yₖ``.

These subclass :class:`aircomp.functions.Operation`, so they run through the same
:func:`aircomp.evaluation.evaluate_operation` harness as the scalar operations.
Iterative operations (e.g. distributed PCA via power iteration) are provided as
functions in the test-bench script rather than single-shot operations.
"""

from __future__ import annotations

import numpy as np

from .functions import Operation


# --------------------------------------------------------------------------- #
# Symmetric-matrix packing (transmit only the upper triangle)
# --------------------------------------------------------------------------- #
def tri_len(dim: int) -> int:
    """Number of upper-triangular entries (incl. diagonal) of a ``dim`` matrix."""
    return dim * (dim + 1) // 2


def pack_sym(matrix: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle (incl. diagonal) of a symmetric matrix."""
    iu = np.triu_indices(matrix.shape[0])
    return matrix[iu]


def unpack_sym(vec: np.ndarray, dim: int) -> np.ndarray:
    """Rebuild a symmetric matrix from its packed upper triangle."""
    iu = np.triu_indices(dim)
    m = np.zeros((dim, dim), dtype=float)
    m[iu] = vec
    return m + m.T - np.diag(np.diag(m))


# --------------------------------------------------------------------------- #
# Vector operations
# --------------------------------------------------------------------------- #
class VectorSum(Operation):
    """Element-wise vector sum ``Σ_k xₖ`` (xₖ ∈ ℝ^dim)."""

    category = "vector"

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.name = f"vector_sum(d={dim})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_agents, self.dim))

    def pre(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float)  # (K, dim)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.ravel(aggregate_sum)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float).sum(axis=0)


class VectorMean(VectorSum):
    """Element-wise vector mean ``(1/K) Σ_k xₖ``."""

    category = "vector"

    def __init__(self, dim: int = 8):
        super().__init__(dim)
        self.name = f"vector_mean(d={dim})"

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.ravel(aggregate_sum) / n_agents

    def true_value(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float).mean(axis=0)


class FederatedAveraging(Operation):
    """Weighted model averaging ``(Σ_k wₖ θₖ) / (Σ_k wₖ)`` (FedAvg step)."""

    category = "vector"

    def __init__(self, dim: int = 8, weights: np.ndarray | None = None):
        self.dim = dim
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.name = f"fed_averaging(d={dim})"

    def _get_weights(self, n: int) -> np.ndarray:
        return self.weights if self.weights is not None else np.linspace(0.5, 1.5, n)

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        # Each row is an agent's local model / update vector.
        return rng.standard_normal((n_agents, self.dim))

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return self._get_weights(len(data))[:, None] * data

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.ravel(aggregate_sum) / float(np.sum(self._get_weights(n_agents)))

    def true_value(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        w = self._get_weights(len(data))
        return (w[:, None] * data).sum(axis=0) / w.sum()


# --------------------------------------------------------------------------- #
# Matrix operations
# --------------------------------------------------------------------------- #
class GramMatrix(Operation):
    """Second-moment (Gram) matrix ``Σ_k xₖ xₖᵀ`` (symmetric)."""

    category = "matrix"

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.name = f"gram_matrix(d={dim})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_agents, self.dim))

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return np.array([pack_sym(np.outer(x, x)) for x in data])  # (K, tri)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return unpack_sym(np.ravel(aggregate_sum), self.dim)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return data.T @ data


class CovarianceMatrix(Operation):
    """Biased sample covariance ``(1/K) Σ (xₖ−μ)(xₖ−μ)ᵀ`` over the air.

    Transmits the packed second moment (``tri`` entries) *and* the sum vector
    (``dim`` entries); the host forms ``R/K − μμᵀ``.
    """

    category = "matrix"

    def __init__(self, dim: int = 5):
        self.dim = dim
        self._tri = tri_len(dim)
        self.name = f"covariance(d={dim})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_agents, self.dim))

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        rows = [np.concatenate([pack_sym(np.outer(x, x)), x]) for x in data]
        return np.array(rows)  # (K, tri + dim)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        agg = np.ravel(aggregate_sum)
        second_moment = unpack_sym(agg[: self._tri], self.dim) / n_agents
        mean = agg[self._tri :] / n_agents
        return second_moment - np.outer(mean, mean)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        return np.cov(np.asarray(data, dtype=float).T, bias=True)


class MatrixSum(Operation):
    """Element-wise matrix aggregation ``Σ_k Mₖ`` (Mₖ ∈ ℝ^{rows×cols})."""

    category = "matrix"

    def __init__(self, rows: int = 4, cols: int = 4):
        self.rows = rows
        self.cols = cols
        self.name = f"matrix_sum({rows}x{cols})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_agents, self.rows, self.cols))

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return data.reshape(len(data), self.rows * self.cols)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.ravel(aggregate_sum).reshape(self.rows, self.cols)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float).sum(axis=0)


class DistributedMatrixVector(Operation):
    """Distributed matrix-vector product ``(Σ_k Aₖ) x`` via ``Σ_k (Aₖ x)``.

    Each agent holds a matrix ``Aₖ`` and the (fixed, shared) vector ``x`` is
    known to all; the wanted product is recovered with ``dim_out`` channel uses.
    """

    category = "matrix"

    def __init__(self, dim_out: int = 4, dim_in: int = 4, seed: int = 0):
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.x = np.random.default_rng(seed).standard_normal(dim_in)
        self.name = f"mat_vec({dim_out}x{dim_in})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_agents, self.dim_out, self.dim_in))

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return data @ self.x  # (K, dim_out): each agent's Aₖ x

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        return np.ravel(aggregate_sum)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float).sum(axis=0) @ self.x


class DistributedLeastSquares(Operation):
    """Least-squares/ridge weights from over-the-air normal equations.

    Each agent holds features ``xₖ`` and label ``yₖ``.  The host aggregates
    ``XᵀX = Σ xₖ xₖᵀ`` (packed) and ``Xᵀy = Σ xₖ yₖ``, then solves
    ``(XᵀX + λI) w = Xᵀy``.  The metric is the error on the recovered weights,
    so it exposes how channel noise propagates through the matrix inverse.
    """

    category = "matrix"

    def __init__(self, dim: int = 4, ridge: float = 1e-6):
        self.dim = dim
        self._tri = tri_len(dim)
        self.ridge = ridge
        self.name = f"least_squares(d={dim})"

    def sample_data(self, n_agents: int, rng: np.random.Generator) -> np.ndarray:
        X = rng.standard_normal((n_agents, self.dim))
        w_true = rng.standard_normal(self.dim)
        y = X @ w_true + 0.1 * rng.standard_normal(n_agents)
        return np.column_stack([X, y])  # (K, dim + 1)

    def pre(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        X, y = data[:, : self.dim], data[:, self.dim]
        rows = [
            np.concatenate([pack_sym(np.outer(X[k], X[k])), X[k] * y[k]])
            for k in range(len(data))
        ]
        return np.array(rows)  # (K, tri + dim)

    def _solve(self, xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
        return np.linalg.solve(xtx + self.ridge * np.eye(self.dim), xty)

    def post(self, aggregate_sum, n_agents: int) -> np.ndarray:
        agg = np.ravel(aggregate_sum)
        xtx = unpack_sym(agg[: self._tri], self.dim)
        xty = agg[self._tri :]
        return self._solve(xtx, xty)

    def true_value(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        X, y = data[:, : self.dim], data[:, self.dim]
        return self._solve(X.T @ X, X.T @ y)


#: A representative linear-algebra operation set for the test bench.
DEFAULT_LINALG_OPERATIONS = [
    VectorSum(dim=8),
    VectorMean(dim=8),
    FederatedAveraging(dim=8),
    GramMatrix(dim=5),
    CovarianceMatrix(dim=5),
    MatrixSum(rows=4, cols=4),
    DistributedMatrixVector(dim_out=4, dim_in=4),
    DistributedLeastSquares(dim=4),
]
