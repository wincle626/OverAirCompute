"""Evaluate every AirComp-supported arithmetic operation.

Runs each operation from the three families through the indoor AirComp channel
and reports its computation accuracy:

* NMSE (dB) of the AirComp estimate vs. the exact value,
* RMSE / MAE (absolute error),
* for the *approximated* operations, the noiseless "approximation floor",
* for the *discrete* operations, the exact-match rate after rounding.

A grouped table is printed and a bar chart of NMSE per operation is saved.

    python examples/evaluate_operations.py
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
    DEFAULT_OPERATIONS,
    OptimizedBeamformingAggregator,
    SystemConfig,
    evaluate_operations,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 300


def main() -> None:
    config = SystemConfig(
        n_agents=40,
        n_hosts=2,
        host_antennas=8,
        agent_antennas=2,
        tx_power_dbm=20.0,
        seed=2025,
    )
    aggregator = OptimizedBeamformingAggregator(n_iter=120)

    print(
        f"Scenario: {config.n_agents} agents x {config.agent_antennas} ant. -> "
        f"{config.n_hosts} hosts x {config.host_antennas} ant., "
        f"{config.tx_power_dbm:.0f} dBm, {N_TRIALS} trials/op\n"
    )

    results = evaluate_operations(
        DEFAULT_OPERATIONS, config, aggregator, n_trials=N_TRIALS
    )

    # Grouped, ordered print-out.
    order = ["natural-medium", "nomographic", "advanced"]
    titles = {
        "natural-medium": "Natural-medium operations",
        "nomographic": "Nomographic (pre-/post-processed) operations",
        "advanced": "Advanced / approximated operations",
    }
    for cat in order:
        print(titles[cat])
        print("-" * len(titles[cat]))
        for r in results:
            if r.category == cat:
                print("  " + r.summary())
        print()

    _bar_chart(results)


def _bar_chart(results) -> None:
    names = [r.name for r in results]
    # Clamp numerically-zero errors so "exact" ops don't dwarf the axis.
    floor = -140.0
    nmse = [max(r.nmse_db, floor) for r in results]
    colors = {
        "natural-medium": "#4C78A8",
        "nomographic": "#59A14F",
        "advanced": "#E15759",
    }
    bar_colors = [colors[r.category] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, nmse, color=bar_colors)
    ax.set_ylabel("Computation NMSE [dB]  (lower = better)")
    ax.set_title("AirComp accuracy per operation (40 agents, 2 hosts x 8 ant.)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=40, ha="right")
    # Mark operations whose error is numerically zero at this SNR.
    for bar, r in zip(bars, results):
        if r.nmse_db < floor:
            ax.text(bar.get_x() + bar.get_width() / 2, floor + 2, "~exact",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), title="family")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "operation_accuracy.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
