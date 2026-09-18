"""Demonstration experiments for the AirComp framework.

Runs three studies and saves the plots next to this script:

1. Computation NMSE vs. per-agent transmit power (SNR), comparing the MRC
   baseline against the optimised aggregation beamformer.
2. NMSE vs. number of agents (with few hosts), illustrating how the worst-agent
   bottleneck of channel-inversion AirComp scales.
3. NMSE vs. receive antennas per host, showing the aggregation-diversity gain.

Run from the project root:

    python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

# Allow running the script directly without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircomp import (  # noqa: E402
    ChannelInversionAggregator,
    OptimizedBeamformingAggregator,
    Simulator,
    SystemConfig,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 300


def _base_config(**overrides) -> SystemConfig:
    cfg = SystemConfig(
        n_agents=40,
        n_hosts=2,
        host_antennas=8,
        tx_power_dbm=10.0,
        seed=2024,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def study_tx_power() -> None:
    powers = np.arange(-10, 31, 5)
    plt.figure(figsize=(7, 4.5))
    for label, agg in [
        ("MRC + channel inversion", ChannelInversionAggregator("mrc")),
        ("Optimised beamforming", OptimizedBeamformingAggregator(n_iter=150)),
    ]:
        sim = Simulator(_base_config(), aggregator=agg)
        res = sim.sweep_tx_power(powers, n_trials=N_TRIALS)
        plt.plot(res.param_values, res.nmse_db, marker="o", label=label)
        print(f"[tx_power] {label}: NMSE(dB) = {np.round(res.nmse_db, 2)}")

    plt.xlabel("Per-agent transmit power [dBm]")
    plt.ylabel("Computation NMSE [dB]")
    plt.title("AirComp accuracy vs. transmit power (40 agents, 2 hosts x 8 ant.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nmse_vs_txpower.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


def study_n_agents() -> None:
    counts = np.array([5, 10, 20, 40, 80, 160])
    plt.figure(figsize=(7, 4.5))
    for label, agg in [
        ("MRC + channel inversion", ChannelInversionAggregator("mrc")),
        ("Optimised beamforming", OptimizedBeamformingAggregator(n_iter=150)),
    ]:
        sim = Simulator(_base_config(), aggregator=agg)
        res = sim.sweep_n_agents(counts, n_trials=N_TRIALS)
        plt.plot(res.param_values, res.nmse_db, marker="s", label=label)
        print(f"[n_agents] {label}: NMSE(dB) = {np.round(res.nmse_db, 2)}")

    plt.xscale("log")
    plt.xlabel("Number of agents")
    plt.ylabel("Computation NMSE [dB]")
    plt.title("AirComp accuracy vs. agent count (2 hosts x 8 ant., 10 dBm)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nmse_vs_agents.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


def study_host_antennas() -> None:
    antennas = np.array([1, 2, 4, 8, 16, 32])
    plt.figure(figsize=(7, 4.5))
    for label, agg in [
        ("MRC + channel inversion", ChannelInversionAggregator("mrc")),
        ("Optimised beamforming", OptimizedBeamformingAggregator(n_iter=150)),
    ]:
        sim = Simulator(_base_config(), aggregator=agg)
        res = sim.sweep_host_antennas(antennas, n_trials=N_TRIALS)
        plt.plot(res.param_values, res.nmse_db, marker="^", label=label)
        print(f"[host_antennas] {label}: NMSE(dB) = {np.round(res.nmse_db, 2)}")

    plt.xscale("log", base=2)
    plt.xlabel("Receive antennas per host")
    plt.ylabel("Computation NMSE [dB]")
    plt.title("AirComp accuracy vs. host antennas (40 agents, 2 hosts, 10 dBm)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nmse_vs_antennas.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


def study_agent_antennas() -> None:
    antennas = np.array([1, 2, 4, 8])
    plt.figure(figsize=(7, 4.5))
    for label, agg in [
        ("MRC + channel inversion", ChannelInversionAggregator("mrc")),
        ("Optimised beamforming", OptimizedBeamformingAggregator(n_iter=150)),
    ]:
        sim = Simulator(_base_config(), aggregator=agg)
        res = sim.sweep_agent_antennas(antennas, n_trials=N_TRIALS)
        plt.plot(res.param_values, res.nmse_db, marker="d", label=label)
        print(f"[agent_antennas] {label}: NMSE(dB) = {np.round(res.nmse_db, 2)}")

    plt.xscale("log", base=2)
    plt.xlabel("Transmit antennas per agent")
    plt.ylabel("Computation NMSE [dB]")
    plt.title("AirComp accuracy vs. agent antennas (40 agents, 2 hosts x 8 ant.)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nmse_vs_agent_antennas.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    study_tx_power()
    study_n_agents()
    study_host_antennas()
    study_agent_antennas()
    print("Done.")
