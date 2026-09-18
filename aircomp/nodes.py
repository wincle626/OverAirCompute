"""Node models: transmitting agents and receiving hosts.

Both agents and hosts may carry antenna arrays; either can be single- or
multi-antenna.  A multi-antenna agent performs transmit beamforming (MRT), while
multi-antenna hosts provide receive dimensions for aggregation beamforming.

Each node owns a uniform linear array (ULA), described by element offsets (in
wavelengths) relative to the array centre; this geometry drives the
line-of-sight steering vectors used by the channel model.  A single-antenna node
is simply a ULA with one element at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import build_ula, dbm_to_watt


@dataclass
class Agent:
    """A wireless agent (edge device / sensor), single- or multi-antenna."""

    node_id: int
    position: np.ndarray  # shape (3,), metres
    n_antennas: int = 1
    spacing_wavelengths: float = 0.5
    tx_power_dbm: float = 20.0
    element_offsets_wavelengths: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        if self.n_antennas < 1:
            raise ValueError("n_antennas must be >= 1")
        self.element_offsets_wavelengths = build_ula(
            self.n_antennas, self.spacing_wavelengths
        )

    @property
    def tx_power_watt(self) -> float:
        return float(dbm_to_watt(self.tx_power_dbm))


@dataclass
class Host:
    """A receiving aggregation host (access point / parameter server).

    ``element_offsets_wavelengths`` has shape ``(n_antennas, 3)`` and stores each
    element's displacement from the array centre in wavelengths, so it can be
    combined directly with a propagation direction to form a steering vector
    (see :func:`aircomp.utils.steering_vector`).
    """

    node_id: int
    position: np.ndarray  # shape (3,), metres
    n_antennas: int = 8
    spacing_wavelengths: float = 0.5
    element_offsets_wavelengths: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        if self.n_antennas < 1:
            raise ValueError("n_antennas must be >= 1")
        self.element_offsets_wavelengths = build_ula(
            self.n_antennas, self.spacing_wavelengths
        )
