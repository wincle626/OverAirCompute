"""Minimal quickstart: one scenario, one Monte-Carlo run.

    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircomp import (  # noqa: E402
    ArithmeticMean,
    OptimizedBeamformingAggregator,
    Simulator,
    SystemConfig,
)

# A room with many agents and a single host. Antenna counts on BOTH ends are
# configurable: set host_antennas / agent_antennas to 1 for single-antenna
# nodes, or > 1 to enable receive / transmit beamforming respectively.
config = SystemConfig(
    n_agents=50,
    n_hosts=1,
    host_antennas=8,   # multi-antenna host  -> receive aggregation beamforming
    agent_antennas=2,  # multi-antenna agents -> transmit (MRT) beamforming
    tx_power_dbm=10.0,
    seed=0,
)

sim = Simulator(
    config,
    aggregator=OptimizedBeamformingAggregator(n_iter=200),
    function=ArithmeticMean(),
)

result = sim.run_monte_carlo(n_trials=500)
print(
    f"{config.n_agents} agents x {config.agent_antennas} ant. -> "
    f"{config.n_hosts} host(s) x {config.host_antennas} ant."
)
print(f"Empirical computation NMSE : {result.nmse_db:6.2f} dB")
print(f"Analytical expected MSE    : {result.mean_expected_mse:.3e}")
