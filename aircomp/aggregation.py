"""AirComp receivers: aggregation beamforming and transmit power control.

MIMO signal model
-----------------
``K`` agents each transmit a standardised symbol ``x_k`` (with ``E|x_k|^2 = 1``)
through a transmit beamforming vector ``b_k`` of length ``N_t`` (the agent's
antenna count), subject to a power budget ``||b_k||^2 <= P_k``.  With the stacked
MIMO channel ``H`` of shape ``(R, K, N_t)`` (``R`` = total receive antennas), the
received signal is

    y = sum_k H_k b_k x_k + n,     n ~ CN(0, sigma^2 I).

A receive combiner ``m`` (length ``R``) and denoising factor ``eta`` produce the
estimate of the target sum ``T = sum_k x_k``:

    T_hat = (m^H y) / sqrt(eta).

For a fixed combiner ``m`` the effective (post-combining) channel of agent ``k``
is the vector ``f_k = H_k^H m`` of length ``N_t``.  To realise ``g_k = m^H H_k
b_k = sqrt(eta)`` with minimum power, the agent uses maximum-ratio transmission

    b_k = sqrt(eta) * f_k / ||f_k||^2,     giving  ||b_k||^2 = eta / ||f_k||^2,

so the power budget forces ``eta = min_k P_k ||f_k||^2 = min_k P_k ||H_k^H m||^2``
(the worst agent limits the whole computation).  The resulting error is

    MSE = ||m||^2 * sigma^2 / eta.

When ``N_t = 1`` this reduces exactly to the single-antenna channel-inversion
case ``||H_k^H m||^2 = |m^H h_k|^2``.

Two combiner designs are provided:

* :class:`ChannelInversionAggregator` — maximum-ratio combiner
  ``m = sum_{k,t} H[:, k, t]`` (a simple baseline), or a fixed user combiner.
* :class:`OptimizedBeamformingAggregator` — maximises the worst-agent gain
  ``min_k ||H_k^H m||^2 / ||m||^2`` via Riemannian gradient ascent on a smooth
  soft-min surrogate, directly minimising the MSE above.  Together with the MRT
  transmit beamformers this performs the alternating-optimal joint transmit /
  receive design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import ChannelRealization
from .config import ConverterConfig
from .utils import as_rng, quantize_sampled, rms


@dataclass
class AggregationDesign:
    """The receiver / transmitter configuration for one channel realisation."""

    combiner: np.ndarray  # m, shape (R,)
    tx_beamformers: np.ndarray  # b, shape (K, N_t)
    denoising: float  # eta
    expected_mse: float  # analytical per-symbol computation MSE


def _effective_channel(H: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Post-combining per-agent channel ``f_k = H_k^H m``, shape ``(K, N_t)``."""
    return np.einsum("rkt,r->kt", H.conj(), m)


class Aggregator:
    """Base class: design a receiver, then estimate the aggregate sum."""

    def design(
        self, channel: ChannelRealization, tx_power_watt: np.ndarray
    ) -> AggregationDesign:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def _uniform_forcing(
        self, H: np.ndarray, m: np.ndarray, noise_power: float, P: np.ndarray
    ) -> AggregationDesign:
        """Given combiner ``m``, apply MRT transmit + channel-inversion control."""
        f = _effective_channel(H, m)  # (K, N_t)
        gain = np.sum(np.abs(f) ** 2, axis=1)  # ||H_k^H m||^2, shape (K,)
        # Denoising limited by the worst per-agent effective gain.
        eta = float(np.min(P * gain))
        # MRT beamformers achieving m^H H_k b_k = sqrt(eta) for every agent.
        safe = np.where(gain > 0, gain, 1.0)
        b = np.sqrt(eta) * f / safe[:, None]
        b[gain <= 0, :] = 0.0
        mse = float(np.linalg.norm(m) ** 2 * noise_power / eta) if eta > 0 else np.inf
        return AggregationDesign(
            combiner=m, tx_beamformers=b, denoising=eta, expected_mse=mse
        )

    def estimate(
        self,
        channel: ChannelRealization,
        symbols: np.ndarray,
        tx_power_watt: np.ndarray,
        rng: np.random.Generator | int | None = None,
        design: AggregationDesign | None = None,
        converters: ConverterConfig | None = None,
    ) -> tuple[complex, AggregationDesign]:
        """Simulate transmission and return ``(T_hat, design)``.

        ``symbols`` are the standardised transmit symbols ``x`` (``E|x|^2=1``).
        ``converters`` optionally applies DAC (transmit) and ADC (receive)
        quantisation; ``None`` models ideal converters.
        """
        rng = as_rng(rng)
        if design is None:
            design = self.design(channel, tx_power_watt)

        H = channel.H
        x = np.asarray(symbols).astype(complex)

        # Transmit DAC: quantise the unit-power information symbol (RMS ~ 1).
        if converters is not None and converters.dac_bits is not None:
            x = quantize_sampled(
                x, converters.dac_bits, converters.dac_clip_sigma,
                converters.dac_oversampling, rng,
            )

        noise = np.sqrt(channel.noise_power / 2.0) * (
            rng.standard_normal(H.shape[0]) + 1j * rng.standard_normal(H.shape[0])
        )
        # y = sum_k H_k b_k x_k + n
        y = np.einsum("rkt,kt->r", H, design.tx_beamformers * x[:, None]) + noise

        # Receive ADC: quantise each antenna's sample around its RMS (AGC).
        if converters is not None and converters.adc_bits is not None:
            full_scale = converters.adc_clip_sigma * rms(y)
            y = quantize_sampled(
                y, converters.adc_bits, full_scale,
                converters.adc_oversampling, rng,
            )

        t_hat = (design.combiner.conj() @ y) / np.sqrt(design.denoising)
        return complex(t_hat), design


class ChannelInversionAggregator(Aggregator):
    """MRC (or fixed) combiner with MRT + channel-inversion power control."""

    def __init__(self, combiner: str | np.ndarray = "mrc"):
        self.combiner = combiner

    def _build_combiner(self, H: np.ndarray) -> np.ndarray:
        if isinstance(self.combiner, np.ndarray):
            m = self.combiner.astype(complex)
        elif self.combiner == "mrc":
            # Coherent sum of all agent channel columns.
            m = H.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unknown combiner '{self.combiner}'.")
        norm = np.linalg.norm(m)
        return m / norm if norm > 0 else m

    def design(
        self, channel: ChannelRealization, tx_power_watt: np.ndarray
    ) -> AggregationDesign:
        P = np.broadcast_to(tx_power_watt, (channel.n_agents,)).astype(float)
        m = self._build_combiner(channel.H)
        return self._uniform_forcing(channel.H, m, channel.noise_power, P)


class OptimizedBeamformingAggregator(Aggregator):
    """Aggregation beamforming that minimises the channel-inversion MSE.

    Maximises the worst-agent effective gain ``min_k ||H_k^H m||^2 / ||m||^2`` on
    the unit sphere using projected gradient ascent on a smooth soft-min
    surrogate.  A lightweight, dependency-free alternative to the SDR/SCA solvers
    common in the literature; it reliably improves on MRC in the many-agent
    regime.  Assumes MRT transmit beamforming at each agent.
    """

    def __init__(
        self,
        n_iter: int = 200,
        step_size: float = 0.2,
        smoothing: float = 6.0,
    ):
        self.n_iter = n_iter
        #: Angular step (radians) for the tangent update on the unit sphere.
        self.step_size = step_size
        #: Soft-min temperature; larger concentrates on the worst agent(s).
        self.smoothing = smoothing

    def _optimize_combiner(self, H: np.ndarray) -> np.ndarray:
        r = H.shape[0]
        # Warm start from MRC.
        m = H.sum(axis=(1, 2))
        norm = np.linalg.norm(m)
        m = m / norm if norm > 0 else np.ones(r, dtype=complex) / np.sqrt(r)

        def worst_gain(vec: np.ndarray) -> float:
            f = _effective_channel(H, vec)
            return float(np.min(np.sum(np.abs(f) ** 2, axis=1)))

        best, best_obj = m.copy(), worst_gain(m)

        for _ in range(self.n_iter):
            f = _effective_channel(H, m)  # (K, N_t), f_k = H_k^H m
            gain = np.sum(np.abs(f) ** 2, axis=1)  # (K,)
            # Scale-invariant soft-min weights over the effective gains.
            ref = np.mean(gain) + 1e-30
            logits = -self.smoothing * gain / ref
            logits -= logits.max()
            w = np.exp(logits)
            w /= w.sum()
            # Euclidean gradient of the soft-min surrogate w.r.t. m*:
            #   sum_k w_k R_k m,  R_k = H_k H_k^H,  R_k m = H_k f_k.
            grad = np.einsum("rkt,kt->r", H, w[:, None] * f)
            # Riemannian (unit-sphere) ascent along the normalised tangent, so
            # progress is independent of the (tiny) channel scale.
            tangent = grad - np.vdot(m, grad) * m
            nt = np.linalg.norm(tangent)
            if nt < 1e-15:
                break
            direction = tangent / nt
            m = np.cos(self.step_size) * m + np.sin(self.step_size) * direction
            m /= np.linalg.norm(m)

            obj = worst_gain(m)
            if obj > best_obj:
                best_obj, best = obj, m.copy()

        return best

    def design(
        self, channel: ChannelRealization, tx_power_watt: np.ndarray
    ) -> AggregationDesign:
        P = np.broadcast_to(tx_power_watt, (channel.n_agents,)).astype(float)
        m = self._optimize_combiner(channel.H)
        return self._uniform_forcing(channel.H, m, channel.noise_power, P)
