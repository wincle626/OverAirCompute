"""End-to-end Monte-Carlo orchestration for AirComp experiments.

The :class:`Simulator` wires together an environment, a channel model, a target
function, and an aggregator, and runs repeated trials to estimate the
computation NMSE.  Convenience sweeps are provided over transmit power (SNR),
number of agents, and number of host antennas.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .aggregation import Aggregator, ChannelInversionAggregator
from .channel import IndoorChannel
from .config import SystemConfig
from .environment import IndoorEnvironment
from .functions import ArithmeticMean, NomographicFunction
from .metrics import nmse, nmse_db
from .utils import as_rng, crandn


@dataclass
class TrialResult:
    """Aggregated outcome of a batch of Monte-Carlo trials."""

    nmse: float
    nmse_db: float
    mean_expected_mse: float
    estimates: np.ndarray = field(repr=False)
    truths: np.ndarray = field(repr=False)


@dataclass
class SweepResult:
    """Outcome of a parameter sweep."""

    param_name: str
    param_values: np.ndarray
    nmse: np.ndarray
    nmse_db: np.ndarray


class Simulator:
    """Runs AirComp experiments for a given system configuration."""

    def __init__(
        self,
        config: SystemConfig,
        aggregator: Aggregator | None = None,
        function: NomographicFunction | None = None,
        environment: IndoorEnvironment | None = None,
    ):
        self.config = config
        self.aggregator = aggregator or ChannelInversionAggregator("mrc")
        self.function = function or ArithmeticMean()
        self.rng = as_rng(config.seed)
        self.environment = environment or IndoorEnvironment(config, rng=self.rng)
        self.channel = IndoorChannel(config.channel, self.environment)

    # ------------------------------------------------------------------ #
    def _tx_power_watt(self) -> np.ndarray:
        return np.array([a.tx_power_watt for a in self.environment.agents])

    def run_monte_carlo(
        self, n_trials: int = 500, resample_topology: bool = False
    ) -> TrialResult:
        """Estimate computation NMSE over ``n_trials`` random realisations.

        The metric is the error on the aggregated sum ``T = sum_k x_k`` of the
        standardised transmit symbols, i.e. the pure computation error of the
        analog aggregation (independent of the downstream function).
        """
        P = self._tx_power_watt()
        estimates = np.empty(n_trials, dtype=complex)
        truths = np.empty(n_trials, dtype=complex)
        mse_acc = 0.0

        for t in range(n_trials):
            if resample_topology:
                self.environment = IndoorEnvironment(self.config, rng=self.rng)
                self.channel = IndoorChannel(self.config.channel, self.environment)
                P = self._tx_power_watt()

            channel = self.channel.realize(self.rng)
            # Standardised transmit symbols (unit power), and their true sum.
            x = crandn(self.config.n_agents, self.rng)
            truths[t] = np.sum(x)

            t_hat, design = self.aggregator.estimate(
                channel, x, P, rng=self.rng, converters=self.config.converters
            )
            estimates[t] = t_hat
            mse_acc += design.expected_mse

        return TrialResult(
            nmse=nmse(estimates, truths),
            nmse_db=nmse_db(estimates, truths),
            mean_expected_mse=mse_acc / n_trials,
            estimates=estimates,
            truths=truths,
        )

    # ------------------------------------------------------------------ #
    def sweep_tx_power(
        self, powers_dbm, n_trials: int = 500, resample_topology: bool = True
    ) -> SweepResult:
        """Sweep per-agent transmit power (a proxy for operating SNR)."""
        powers_dbm = np.atleast_1d(powers_dbm).astype(float)
        out = np.empty(len(powers_dbm))
        for i, p in enumerate(powers_dbm):
            cfg = replace(self.config, tx_power_dbm=float(p))
            sim = self._clone_with_config(cfg)
            out[i] = sim.run_monte_carlo(n_trials, resample_topology).nmse
        return SweepResult("tx_power_dbm", powers_dbm, out, 10 * np.log10(out))

    def sweep_n_agents(
        self, agent_counts, n_trials: int = 500, resample_topology: bool = True
    ) -> SweepResult:
        """Sweep the number of agents (host count held fixed and small)."""
        agent_counts = np.atleast_1d(agent_counts).astype(int)
        out = np.empty(len(agent_counts))
        for i, k in enumerate(agent_counts):
            cfg = replace(self.config, n_agents=int(k))
            sim = self._clone_with_config(cfg)
            out[i] = sim.run_monte_carlo(n_trials, resample_topology).nmse
        return SweepResult("n_agents", agent_counts, out, 10 * np.log10(out))

    def sweep_host_antennas(
        self, antenna_counts, n_trials: int = 500, resample_topology: bool = True
    ) -> SweepResult:
        """Sweep receive antennas per host (aggregation diversity)."""
        antenna_counts = np.atleast_1d(antenna_counts).astype(int)
        out = np.empty(len(antenna_counts))
        for i, n in enumerate(antenna_counts):
            cfg = replace(self.config, host_antennas=int(n))
            sim = self._clone_with_config(cfg)
            out[i] = sim.run_monte_carlo(n_trials, resample_topology).nmse
        return SweepResult("host_antennas", antenna_counts, out, 10 * np.log10(out))

    def sweep_agent_antennas(
        self, antenna_counts, n_trials: int = 500, resample_topology: bool = True
    ) -> SweepResult:
        """Sweep transmit antennas per agent (transmit-beamforming gain)."""
        antenna_counts = np.atleast_1d(antenna_counts).astype(int)
        out = np.empty(len(antenna_counts))
        for i, n in enumerate(antenna_counts):
            cfg = replace(self.config, agent_antennas=int(n))
            sim = self._clone_with_config(cfg)
            out[i] = sim.run_monte_carlo(n_trials, resample_topology).nmse
        return SweepResult("agent_antennas", antenna_counts, out, 10 * np.log10(out))

    # ------------------------------------------------------------------ #
    def _clone_with_config(self, cfg: SystemConfig) -> "Simulator":
        return Simulator(cfg, aggregator=self.aggregator, function=self.function)
