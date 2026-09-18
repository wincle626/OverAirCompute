"""Benchmark AirComp operations against their exact values.

For each Monte-Carlo trial an operation supplies a dataset and its exact result.
The per-agent pre-processed values are standardised to unit power, transmitted
over the AirComp channel (one channel use per output dimension, sharing the same
channel realisation and beamformer), and the received sums are un-standardised
and post-processed into an estimate.  The estimate is compared with the exact
value; for the approximated operations we additionally report the *noiseless*
AirComp value (perfect sum) so the inherent approximation floor is separated
from the channel-induced error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aggregation import Aggregator, ChannelInversionAggregator
from .channel import IndoorChannel
from .config import SystemConfig
from .environment import IndoorEnvironment
from .functions import Operation, standardize
from .metrics import nmse, nmse_db
from .utils import as_rng


@dataclass
class OperationResult:
    """Accuracy summary for one operation over many trials."""

    name: str
    category: str
    approximate: bool
    discrete: bool
    nmse_db: float          # AirComp estimate vs exact value
    rmse: float             # root-mean-square absolute error
    mae: float              # mean absolute error
    approx_nmse_db: float   # noiseless AirComp vs exact (approximation floor)
    success_rate: float     # exact-match rate after rounding (discrete ops only)

    @staticmethod
    def _fmt_db(value: float) -> str:
        # Below ~-150 dB the error is numerically zero (noise-free recovery).
        if not np.isfinite(value) or value < -150.0:
            return "  ~exact"
        return f"{value:7.2f} dB"

    def summary(self) -> str:
        base = (
            f"{self.name:<18s} [{self.category:<14s}] "
            f"NMSE={self._fmt_db(self.nmse_db)}  RMSE={self.rmse:.3e}"
        )
        if self.approximate:
            base += f"  approx-floor={self._fmt_db(self.approx_nmse_db)}"
        if self.discrete:
            base += f"  exact-match={100 * self.success_rate:5.1f}%"
        return base


def _aircomp_sum(aggregator, channel, design, g, tx_power, rng, converters=None):
    """Transmit one standardised real vector ``g`` and estimate ``sum_k g_k``."""
    std = standardize(g)
    if std.scale == 1.0 and np.allclose(g, g[0]):
        # Degenerate constant vector carries no information to aggregate.
        return float(np.sum(g))
    x = std.forward(g).astype(complex)
    t_hat, _ = aggregator.estimate(
        channel, x, tx_power, rng=rng, design=design, converters=converters
    )
    return std.invert_sum(t_hat, len(g))


def evaluate_operation(
    operation: Operation,
    config: SystemConfig,
    aggregator: Aggregator | None = None,
    n_trials: int = 300,
    rng: np.random.Generator | int | None = None,
    resample_topology: bool = True,
) -> OperationResult:
    """Estimate the accuracy of one operation under the given scenario."""
    rng = as_rng(rng if rng is not None else config.seed)
    aggregator = aggregator or ChannelInversionAggregator("mrc")

    env = IndoorEnvironment(config, rng=rng)
    chan = IndoorChannel(config.channel, env)
    tx_power = np.array([a.tx_power_watt for a in env.agents])

    estimates, truths, approx_vals = [], [], []
    hits = 0

    for _ in range(n_trials):
        if resample_topology:
            env = IndoorEnvironment(config, rng=rng)
            chan = IndoorChannel(config.channel, env)
            tx_power = np.array([a.tx_power_watt for a in env.agents])

        data = operation.sample_data(config.n_agents, rng)
        n = len(np.asarray(data))
        pre = operation._pre_2d(data)  # (K, D)
        n_dims = pre.shape[1]

        channel = chan.realize(rng)
        design = aggregator.design(channel, tx_power)

        agg_hat = np.array(
            [
                _aircomp_sum(
                    aggregator, channel, design, pre[:, d], tx_power, rng,
                    config.converters,
                )
                for d in range(n_dims)
            ]
        )

        est = np.ravel(operation.post(agg_hat, n)).astype(float)
        truth = np.ravel(np.asarray(operation.true_value(data), dtype=float))
        approx = np.ravel(
            np.asarray(operation.post(pre.sum(axis=0), n), dtype=float)
        )

        estimates.append(est)
        truths.append(truth)
        approx_vals.append(approx)
        if operation.discrete and np.array_equal(np.round(est), np.round(truth)):
            hits += 1

    estimates = np.concatenate(estimates)
    truths = np.concatenate(truths)
    approx_vals = np.concatenate(approx_vals)
    errors = estimates - truths

    return OperationResult(
        name=operation.name,
        category=operation.category,
        approximate=operation.approximate,
        discrete=operation.discrete,
        nmse_db=nmse_db(estimates, truths),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mae=float(np.mean(np.abs(errors))),
        approx_nmse_db=nmse_db(approx_vals, truths) if operation.approximate else float("-inf"),
        success_rate=(hits / n_trials) if operation.discrete else float("nan"),
    )


def evaluate_operations(
    operations,
    config: SystemConfig,
    aggregator: Aggregator | None = None,
    n_trials: int = 300,
    rng: np.random.Generator | int | None = None,
    resample_topology: bool = True,
) -> list[OperationResult]:
    """Evaluate a list of operations under a shared scenario."""
    rng = as_rng(rng if rng is not None else config.seed)
    results = []
    for op in operations:
        # Give each operation an independent, reproducible stream.
        sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        results.append(
            evaluate_operation(op, config, aggregator, n_trials, sub_rng, resample_topology)
        )
    return results
