"""Indoor wireless channel model for (MIMO) AirComp.

Each host-agent link is a matrix ``H_k`` mapping the agent's ``N_t`` transmit
antennas to the host's receive antennas:

    H_k = sqrt(beta) * ( sqrt(K/(K+1)) a_rx a_tx^H + sqrt(1/(K+1)) G )

where ``beta`` is the large-scale gain from a log-distance path-loss model with
log-normal shadowing, ``a_rx`` / ``a_tx`` are geometry-derived line-of-sight
(LoS) steering vectors of the host and agent arrays, and ``G`` is an i.i.d.
Rayleigh (NLoS) component.  The LoS term is the rank-one outer product
``a_rx a_tx^H``, i.e. a plane wave leaving the agent array in the host's
direction and arriving at the host array from the agent's direction.

For ``H`` hosts of ``N_r`` antennas, ``K`` agents of ``N_t`` antennas, the
channel realisation is stored as a single array of shape
``(H*N_r, K, N_t)`` — all receive antennas stacked, so a central processor can
perform *joint* aggregation beamforming across cooperating hosts (cell-free
style AirComp).  Single-antenna nodes are the special cases ``N_r = 1`` and/or
``N_t = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ChannelConfig
from .environment import IndoorEnvironment
from .utils import (
    SPEED_OF_LIGHT,
    as_rng,
    crandn,
    db_to_linear,
    dbm_to_watt,
    steering_vector,
    unit_vector,
)


@dataclass
class ChannelRealization:
    """One realisation of the (stacked) uplink MIMO channel.

    Attributes
    ----------
    H:
        Complex channel tensor of shape ``(total_rx_antennas, n_agents,
        agent_antennas)``.  ``H[:, k, :]`` is the ``R x N_t`` matrix of agent k.
    noise_power:
        Receiver noise power per antenna [W].
    large_scale_gain:
        Per-link large-scale power gain ``beta``, shape ``(n_hosts, n_agents)``.
    """

    H: np.ndarray
    noise_power: float
    large_scale_gain: np.ndarray

    @property
    def n_rx(self) -> int:
        return self.H.shape[0]

    @property
    def n_agents(self) -> int:
        return self.H.shape[1]

    @property
    def n_tx(self) -> int:
        return self.H.shape[2]


class IndoorChannel:
    """Generates channel realisations for a fixed environment topology."""

    def __init__(self, config: ChannelConfig, environment: IndoorEnvironment):
        self.config = config
        self.env = environment
        self.wavelength = SPEED_OF_LIGHT / config.carrier_freq_hz
        self.noise_power = float(dbm_to_watt(config.noise_power_dbm()))

        # Path-loss intercept at the reference distance (free-space).
        self._pl0_db = 20.0 * np.log10(
            4.0 * np.pi * config.ref_distance_m / self.wavelength
        )
        # Topology is fixed for a given environment; only fading and shadowing
        # are random. Precompute distances and the LoS transmit/receive
        # steering vectors for every host-agent pair.
        self._distances = self.env.link_distances()  # (H, K)
        self._los_rx, self._los_tx = self._precompute_steering()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _precompute_steering(self):
        """LoS steering vectors, indexed as ``[host][agent]``.

        ``rx`` is the host-array response (arrival direction), ``tx`` is the
        agent-array response (departure direction, i.e. towards the host).
        """
        rx: list[list[np.ndarray]] = []
        tx: list[list[np.ndarray]] = []
        for host in self.env.hosts:
            rx_h, tx_h = [], []
            for agent in self.env.agents:
                d_rx = unit_vector(agent.position - host.position)  # host -> agent
                rx_h.append(steering_vector(host.element_offsets_wavelengths, d_rx))
                tx_h.append(
                    steering_vector(agent.element_offsets_wavelengths, -d_rx)
                )
            rx.append(rx_h)
            tx.append(tx_h)
        return rx, tx

    def path_loss_db(self, shadowing_db: np.ndarray | None = None) -> np.ndarray:
        """Log-distance path loss [dB], shape ``(n_hosts, n_agents)``."""
        d = np.maximum(self._distances, self.config.ref_distance_m)
        pl = self._pl0_db + 10.0 * self.config.path_loss_exponent * np.log10(
            d / self.config.ref_distance_m
        )
        if shadowing_db is not None:
            pl = pl + shadowing_db
        return pl

    # ------------------------------------------------------------------ #
    # Realisation
    # ------------------------------------------------------------------ #
    def realize(
        self, rng: np.random.Generator | int | None = None
    ) -> ChannelRealization:
        """Draw one channel realisation (fresh shadowing + small-scale fading)."""
        rng = as_rng(rng)
        n_hosts = len(self.env.hosts)
        n_agents = len(self.env.agents)
        n_rx = self.env.config.host_antennas
        n_tx = self.env.config.agent_antennas

        # Large-scale gain beta = 10^(-PL/10) with per-link log-normal shadowing.
        shadowing = rng.normal(
            0.0, self.config.shadowing_std_db, size=(n_hosts, n_agents)
        )
        beta = db_to_linear(-self.path_loss_db(shadowing))  # (H, K)

        # Rician small-scale fading.
        k_linear = db_to_linear(self.config.rician_k_db)
        los_scale = np.sqrt(k_linear / (k_linear + 1.0))
        nlos_scale = np.sqrt(1.0 / (k_linear + 1.0))

        H = np.zeros((n_hosts * n_rx, n_agents, n_tx), dtype=complex)
        for h_idx in range(n_hosts):
            row0 = h_idx * n_rx
            for k in range(n_agents):
                a_rx = self._los_rx[h_idx][k]  # (n_rx,)
                a_tx = self._los_tx[h_idx][k]  # (n_tx,)
                los = np.outer(a_rx, np.conj(a_tx))  # (n_rx, n_tx)
                nlos = crandn((n_rx, n_tx), rng)
                block = los_scale * los + nlos_scale * nlos
                H[row0 : row0 + n_rx, k, :] = np.sqrt(beta[h_idx, k]) * block

        return ChannelRealization(
            H=H, noise_power=self.noise_power, large_scale_gain=beta
        )
