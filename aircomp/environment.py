"""Indoor environment: room geometry and node placement.

The environment is a shoebox room.  Agents are dropped uniformly at random
inside it; hosts are, by default, mounted near the ceiling in a regular pattern
(a single host is placed at the ceiling centre), reflecting typical indoor
access-point deployments.  Custom placements can be supplied explicitly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import SystemConfig
from .nodes import Agent, Host
from .utils import as_rng


class IndoorEnvironment:
    """Holds the room, the agents, and the hosts for one scenario topology."""

    def __init__(
        self,
        config: SystemConfig,
        agents: Sequence[Agent] | None = None,
        hosts: Sequence[Host] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.config = config
        self.rng = as_rng(rng if rng is not None else config.seed)
        self.hosts = list(hosts) if hosts is not None else self._place_hosts()
        self.agents = list(agents) if agents is not None else self._place_agents()

    # ------------------------------------------------------------------ #
    # Placement
    # ------------------------------------------------------------------ #
    def _place_agents(self) -> list[Agent]:
        room = self.config.room
        n = self.config.n_agents
        xs = self.rng.uniform(0.0, room.length, size=n)
        ys = self.rng.uniform(0.0, room.width, size=n)
        # Devices sit between the floor and ~1.5 m (desk / hand height).
        zs = self.rng.uniform(0.0, min(1.5, room.height), size=n)
        return [
            Agent(
                node_id=i,
                position=np.array([xs[i], ys[i], zs[i]]),
                n_antennas=self.config.agent_antennas,
                spacing_wavelengths=self.config.array_spacing_wavelengths,
                tx_power_dbm=self.config.tx_power_dbm,
            )
            for i in range(n)
        ]

    def _place_hosts(self) -> list[Host]:
        room = self.config.room
        n = self.config.n_hosts
        z = room.height  # ceiling-mounted

        if n == 1:
            centres = [np.array([room.length / 2.0, room.width / 2.0, z])]
        else:
            # Arrange hosts on a near-square grid across the ceiling.
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            centres = []
            for r in range(rows):
                for c in range(cols):
                    if len(centres) >= n:
                        break
                    x = room.length * (c + 0.5) / cols
                    y = room.width * (r + 0.5) / rows
                    centres.append(np.array([x, y, z]))

        return [
            Host(
                node_id=i,
                position=centres[i],
                n_antennas=self.config.host_antennas,
                spacing_wavelengths=self.config.array_spacing_wavelengths,
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def agent_positions(self) -> np.ndarray:
        """Return agent positions stacked as an ``(n_agents, 3)`` array."""
        return np.array([a.position for a in self.agents])

    def host_positions(self) -> np.ndarray:
        """Return host centres stacked as an ``(n_hosts, 3)`` array."""
        return np.array([h.position for h in self.hosts])

    def link_distances(self) -> np.ndarray:
        """Euclidean host-agent distances, shape ``(n_hosts, n_agents)`` [m]."""
        hp = self.host_positions()[:, None, :]  # (H, 1, 3)
        ap = self.agent_positions()[None, :, :]  # (1, K, 3)
        return np.linalg.norm(hp - ap, axis=2)
