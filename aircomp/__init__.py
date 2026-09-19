"""AirComp: an over-the-air computation (AirComp) simulation framework.

This package simulates analog over-the-air computation in an indoor wireless
environment, where a small number of multi-antenna *hosts* (access points /
parameter servers) aggregate signals transmitted concurrently by a large number
of single-antenna *agents* (sensors / edge devices).

The design exploits the signal-superposition property of the wireless
multiple-access channel: when all agents transmit simultaneously, the host
receives the (channel-weighted) *sum* of their signals, which — after suitable
pre-/post-processing — realises a nomographic function such as the arithmetic
mean, weighted sum, or geometric mean.

Typical entry points
--------------------
>>> from aircomp import Simulator, SystemConfig
>>> sim = Simulator(SystemConfig(n_agents=50, n_hosts=1, host_antennas=8))
>>> result = sim.run_monte_carlo(n_trials=200)
>>> print(result.nmse_db)

Sub-modules
-----------
config       : dataclass parameter containers.
environment  : indoor room geometry and node placement.
nodes        : Agent and Host models (including antenna arrays).
channel      : indoor path-loss + Rician fading channel generator.
functions    : nomographic target functions (mean, sum, ...).
aggregation  : AirComp receivers (beamforming + power control).
metrics      : MSE / NMSE computation.
simulator    : orchestrates end-to-end Monte-Carlo experiments.
"""

from .config import ChannelConfig, ConverterConfig, RoomConfig, SystemConfig
from .environment import IndoorEnvironment
from .nodes import Agent, Host
from .channel import IndoorChannel, ChannelRealization
from .functions import (
    DEFAULT_OPERATIONS,
    ArithmeticMean,
    Counting,
    EuclideanNorm,
    GeometricMean,
    Histogram,
    MajorityVote,
    MaxApprox,
    MinApprox,
    NomographicFunction,
    Operation,
    PolynomialFunction,
    Subtraction,
    Summation,
    WeightedAverage,
    WeightedSum,
    standardize,
)
from .aggregation import (
    Aggregator,
    ChannelInversionAggregator,
    OptimizedBeamformingAggregator,
)
from .linalg import (
    DEFAULT_LINALG_OPERATIONS,
    CovarianceMatrix,
    DistributedLeastSquares,
    DistributedMatrixVector,
    FederatedAveraging,
    GramMatrix,
    MatrixSum,
    VectorMean,
    VectorSum,
)
from .metrics import mse, nmse, nmse_db
from .evaluation import (
    OperationResult,
    aircomp_vector,
    evaluate_operation,
    evaluate_operations,
)
from .simulator import Simulator, SweepResult, TrialResult

__version__ = "0.1.0"

__all__ = [
    "ChannelConfig",
    "ConverterConfig",
    "RoomConfig",
    "SystemConfig",
    "IndoorEnvironment",
    "Agent",
    "Host",
    "IndoorChannel",
    "ChannelRealization",
    "Operation",
    "NomographicFunction",
    "Summation",
    "Subtraction",
    "ArithmeticMean",
    "WeightedSum",
    "WeightedAverage",
    "GeometricMean",
    "EuclideanNorm",
    "PolynomialFunction",
    "MajorityVote",
    "Counting",
    "Histogram",
    "MaxApprox",
    "MinApprox",
    "DEFAULT_OPERATIONS",
    "standardize",
    "VectorSum",
    "VectorMean",
    "FederatedAveraging",
    "GramMatrix",
    "CovarianceMatrix",
    "MatrixSum",
    "DistributedMatrixVector",
    "DistributedLeastSquares",
    "DEFAULT_LINALG_OPERATIONS",
    "aircomp_vector",
    "Aggregator",
    "ChannelInversionAggregator",
    "OptimizedBeamformingAggregator",
    "mse",
    "nmse",
    "nmse_db",
    "OperationResult",
    "evaluate_operation",
    "evaluate_operations",
    "Simulator",
    "SweepResult",
    "TrialResult",
]
