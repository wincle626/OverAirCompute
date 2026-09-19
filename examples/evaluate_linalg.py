"""Test bench for vector/matrix linear-algebra operations over AirComp.

Part 1 — single-shot operations (element-wise vector/matrix aggregation,
federated averaging, Gram/covariance matrices, distributed matrix-vector
product, distributed least squares) evaluated for computation NMSE.

Part 2 — an iterative example: distributed PCA by power iteration, where each
iteration computes ``C v = (1/K) Σ_k xₖ (xₖᵀ v)`` over the air.  We report how
well the recovered top eigenvector aligns with the true one.

    python examples/evaluate_linalg.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircomp import (  # noqa: E402
    DEFAULT_LINALG_OPERATIONS,
    IndoorChannel,
    IndoorEnvironment,
    OptimizedBeamformingAggregator,
    SystemConfig,
    aircomp_vector,
    evaluate_operations,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 200


def _config() -> SystemConfig:
    return SystemConfig(
        n_agents=40,
        n_hosts=2,
        host_antennas=8,
        agent_antennas=2,
        tx_power_dbm=20.0,
        seed=2026,
    )


# --------------------------------------------------------------------------- #
# Part 1 — single-shot operations
# --------------------------------------------------------------------------- #
def run_single_shot() -> None:
    config = _config()
    aggregator = OptimizedBeamformingAggregator(n_iter=120)
    print(
        f"Scenario: {config.n_agents} agents x {config.agent_antennas} ant. -> "
        f"{config.n_hosts} hosts x {config.host_antennas} ant., "
        f"{config.tx_power_dbm:.0f} dBm, {N_TRIALS} trials/op\n"
    )
    print("Vector / matrix linear-algebra operations")
    print("-----------------------------------------")
    results = evaluate_operations(
        DEFAULT_LINALG_OPERATIONS, config, aggregator, n_trials=N_TRIALS
    )
    for r in results:
        print("  " + r.summary())
    print()
    _bar_chart(results)


def _bar_chart(results) -> None:
    names = [r.name for r in results]
    nmse = [max(r.nmse_db, -140.0) for r in results]
    colors = {"vector": "#4C78A8", "matrix": "#59A14F"}
    bar_colors = [colors.get(r.category, "#888") for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, nmse, color=bar_colors)
    ax.set_ylabel("Computation NMSE [dB]  (lower = better)")
    ax.set_title("AirComp linear-algebra accuracy (40 agents, 2 hosts x 8 ant.)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=40, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), title="operand")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "linalg_accuracy.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Part 2 — iterative: distributed PCA by power iteration
# --------------------------------------------------------------------------- #
def run_distributed_pca(dim: int = 5, power_iters: int = 8, n_trials: int = 60) -> None:
    config = _config()
    aggregator = OptimizedBeamformingAggregator(n_iter=120)
    rng = np.random.default_rng(config.seed)

    alignments, ideal_alignments = [], []
    for _ in range(n_trials):
        env = IndoorEnvironment(config, rng=rng)
        chan = IndoorChannel(config.channel, env)
        tx_power = np.array([a.tx_power_watt for a in env.agents])

        # Data with a planted dominant direction.
        u = rng.standard_normal(dim)
        u /= np.linalg.norm(u)
        latent = rng.standard_normal((config.n_agents, 1))
        X = 3.0 * latent * u + rng.standard_normal((config.n_agents, dim))

        C = (X.T @ X) / config.n_agents
        w_true = np.linalg.eigh(C)[1][:, -1]  # true top eigenvector

        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        v_ideal = v.copy()
        for _ in range(power_iters):
            # Over-the-air: Σ_k (xₖᵀ v) xₖ  =  K · C v
            channel = chan.realize(rng)
            design = aggregator.design(channel, tx_power)
            contrib = (X @ v)[:, None] * X  # (K, dim)
            agg = aircomp_vector(aggregator, channel, design, contrib, tx_power, rng)
            v = agg / np.linalg.norm(agg)
            # Noise-free reference power iteration for comparison.
            v_ideal = C @ v_ideal
            v_ideal /= np.linalg.norm(v_ideal)

        alignments.append(abs(v @ w_true))
        ideal_alignments.append(abs(v_ideal @ w_true))

    print("Distributed PCA (power iteration over the air)")
    print("----------------------------------------------")
    print(f"  dim={dim}, power iterations={power_iters}, trials={n_trials}")
    print(f"  |<v_hat, v*>| over-the-air : {np.mean(alignments):.4f} "
          f"(1.0 = perfect eigenvector recovery)")
    print(f"  |<v_hat, v*>| noise-free   : {np.mean(ideal_alignments):.4f}  (algorithm floor)")
    print()


if __name__ == "__main__":
    run_single_shot()
    run_distributed_pca()
    print("Done.")
