"""Configuration dataclasses for the AirComp simulator.

Keeping every tunable parameter in typed dataclasses makes experiments
reproducible and self-documenting; a whole scenario is fully described by a
``SystemConfig`` (which nests a ``RoomConfig`` and a ``ChannelConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoomConfig:
    """Dimensions of the (shoebox) indoor environment, in metres."""

    length: float = 12.0
    width: float = 8.0
    height: float = 3.0

    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class ChannelConfig:
    """Indoor propagation and noise parameters.

    Defaults correspond to a 2.4 GHz indoor office at room temperature with a
    mild line-of-sight component, following commonly used log-distance and
    Rician models (e.g. ITU-R P.1238 style path loss).
    """

    carrier_freq_hz: float = 2.4e9
    #: Log-distance path-loss exponent (indoor office ~ 1.6-3.5).
    path_loss_exponent: float = 2.2
    #: Reference distance for the path-loss intercept [m].
    ref_distance_m: float = 1.0
    #: Log-normal shadow-fading standard deviation [dB].
    shadowing_std_db: float = 4.0
    #: Rician K-factor [dB]; large -> strong LoS, ``-inf`` -> Rayleigh.
    rician_k_db: float = 6.0
    #: Signal bandwidth [Hz], used to compute the thermal noise power.
    bandwidth_hz: float = 1.0e6
    #: Receiver noise figure [dB].
    noise_figure_db: float = 7.0
    #: Thermal noise power spectral density [dBm/Hz] (kTB at ~290 K).
    noise_psd_dbm_hz: float = -174.0

    def noise_power_dbm(self) -> float:
        """Total receiver noise power over the signal bandwidth [dBm]."""
        import numpy as np

        return (
            self.noise_psd_dbm_hz
            + 10.0 * np.log10(self.bandwidth_hz)
            + self.noise_figure_db
        )


@dataclass
class ConverterConfig:
    """Optional DAC/ADC quantisation model.

    Data converters are the analog-domain bottleneck of AirComp.  With ``bits``
    set to ``None`` the converter is ideal (no quantisation, the default, so the
    channel-noise-limited results are unchanged).  Setting a finite resolution
    adds uniform quantisation with clipping: the full-scale range is placed at
    ``clip_sigma`` times the signal RMS (a simple automatic-gain-control model),
    so a small ``clip_sigma`` clips the peaks while a large one wastes resolution.
    """

    #: Transmit DAC amplitude resolution [bits]; ``None`` -> ideal DAC.
    dac_bits: int | None = None
    #: Receive ADC amplitude resolution [bits]; ``None`` -> ideal ADC.
    adc_bits: int | None = None
    #: DAC full-scale peak, in units of the transmit-symbol RMS.
    dac_clip_sigma: float = 4.0
    #: ADC full-scale peak, in units of the received-signal RMS.
    adc_clip_sigma: float = 4.0
    #: DAC oversampling ratio (samples per symbol); >1 trades sampling rate for
    #: in-band quantisation-noise reduction of ~10*log10(OSR) dB.
    dac_oversampling: int = 1
    #: ADC oversampling ratio (samples per symbol); same processing-gain trade-off.
    adc_oversampling: int = 1

    @property
    def enabled(self) -> bool:
        return self.dac_bits is not None or self.adc_bits is not None


@dataclass
class SystemConfig:
    """Top-level scenario description.

    In AirComp deployments the number of hosts (aggregation points) is much
    smaller than the number of agents; the defaults reflect that regime.
    """

    n_agents: int = 50
    n_hosts: int = 1
    #: Receive antennas per host. Set to 1 for single-antenna hosts.
    host_antennas: int = 8
    #: Transmit antennas per agent. Set to 1 for single-antenna agents;
    #: values > 1 enable transmit beamforming (MRT) at each agent.
    agent_antennas: int = 1
    #: Per-agent maximum transmit power [dBm].
    tx_power_dbm: float = 20.0
    #: Antenna element spacing for the (host and agent) ULAs, in wavelengths.
    array_spacing_wavelengths: float = 0.5
    #: Master random seed for reproducibility.
    seed: int | None = 0

    room: RoomConfig = field(default_factory=RoomConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    converters: ConverterConfig = field(default_factory=ConverterConfig)

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if self.n_hosts < 1:
            raise ValueError("n_hosts must be >= 1")
        if self.host_antennas < 1:
            raise ValueError("host_antennas must be >= 1")
        if self.agent_antennas < 1:
            raise ValueError("agent_antennas must be >= 1")

    @property
    def total_rx_antennas(self) -> int:
        """Total receive dimensions available to the joint aggregator."""
        return self.n_hosts * self.host_antennas
