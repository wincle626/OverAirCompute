"""Small numerical helpers shared across the framework.

Unit conventions
----------------
* Powers are handled internally in **watts**; user-facing configuration is in
  dBm (transmit power, noise power spectral density).
* Distances are in **metres**, frequencies in **Hz**.
* Channel coefficients are dimensionless complex voltage gains such that the
  received power of a symbol with unit power is ``|h|**2``.
"""

from __future__ import annotations

import numpy as np

#: Speed of light in vacuum [m/s].
SPEED_OF_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- #
# dB / linear conversions
# --------------------------------------------------------------------------- #
def db_to_linear(value_db: float | np.ndarray) -> float | np.ndarray:
    """Convert a power ratio in dB to linear scale."""
    return 10.0 ** (np.asarray(value_db) / 10.0)


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a linear power ratio to dB (clipped away from ``log10(0)``)."""
    value = np.asarray(value, dtype=float)
    return 10.0 * np.log10(np.maximum(value, 1e-300))


def dbm_to_watt(power_dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert power in dBm to watts."""
    return 10.0 ** (np.asarray(power_dbm) / 10.0) * 1e-3


def watt_to_dbm(power_watt: float | np.ndarray) -> float | np.ndarray:
    """Convert power in watts to dBm."""
    power_watt = np.asarray(power_watt, dtype=float)
    return 10.0 * np.log10(np.maximum(power_watt, 1e-300) / 1e-3)


# --------------------------------------------------------------------------- #
# Random number generation
# --------------------------------------------------------------------------- #
def as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy ``Generator`` from a seed, generator, or ``None``."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def crandn(shape, rng: np.random.Generator) -> np.ndarray:
    """Draw circularly-symmetric complex Gaussian samples ~ CN(0, 1).

    Each sample has unit total variance (variance 1/2 per real dimension).
    """
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag) / np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Geometry / array processing
# --------------------------------------------------------------------------- #
def quantize(values: np.ndarray, bits: int | None, full_scale: float) -> np.ndarray:
    """Uniform mid-tread quantiser with clipping to ``[-full_scale, full_scale]``.

    ``bits`` is the converter resolution (``2**bits`` levels); ``None`` or a
    non-positive value returns the input unchanged (ideal converter).  Complex
    inputs are quantised on the I and Q rails independently.
    """
    if bits is None or bits <= 0 or full_scale <= 0:
        return values
    values = np.asarray(values)
    if np.iscomplexobj(values):
        return quantize(values.real, bits, full_scale) + 1j * quantize(
            values.imag, bits, full_scale
        )
    step = 2.0 * full_scale / (2 ** int(bits) - 1)
    clipped = np.clip(values, -full_scale, full_scale)
    return np.round(clipped / step) * step


def quantize_sampled(
    values: np.ndarray,
    bits: int | None,
    full_scale: float,
    oversampling: int = 1,
    rng: "np.random.Generator | int | None" = None,
) -> np.ndarray:
    """Quantise with an oversampling (sampling-rate) processing gain.

    With ``oversampling = 1`` this is the plain :func:`quantize`.  With a larger
    oversampling ratio (samples per symbol) the value is quantised ``OSR`` times
    under independent rectangular dither and averaged, modelling how oversampling
    plus matched filtering rejects out-of-band quantisation noise: the in-band
    quantisation-error power falls by roughly ``OSR`` (~3 dB per doubling).
    """
    if bits is None or bits <= 0 or full_scale <= 0:
        return values
    if oversampling is None or oversampling <= 1:
        return quantize(values, bits, full_scale)

    rng = as_rng(rng)
    values = np.asarray(values)
    complex_input = np.iscomplexobj(values)
    step = 2.0 * full_scale / (2 ** int(bits) - 1)
    acc = np.zeros(values.shape, dtype=complex if complex_input else float)
    for _ in range(int(oversampling)):
        dither = rng.uniform(-step / 2.0, step / 2.0, size=values.shape)
        if complex_input:
            dither = dither + 1j * rng.uniform(-step / 2.0, step / 2.0, size=values.shape)
        acc += quantize(values + dither, bits, full_scale)
    return acc / int(oversampling)


def rms(values: np.ndarray) -> float:
    """Root-mean-square magnitude of a (possibly complex) array."""
    values = np.asarray(values)
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit Euclidean norm (safe for near-zero)."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        out = np.zeros_like(vec, dtype=float)
        out[0] = 1.0
        return out
    return vec / norm


def build_ula(n_antennas: int, spacing_wavelengths: float) -> np.ndarray:
    """Element offsets of a uniform linear array along the local x-axis.

    Returns an ``(n_antennas, 3)`` array of positions (in wavelengths) relative
    to the array centre.  A single-antenna array is a single element at the
    origin, so its steering vector is always ``[1]``.
    """
    if n_antennas < 1:
        raise ValueError("n_antennas must be >= 1")
    idx = np.arange(n_antennas)
    offsets = np.zeros((n_antennas, 3), dtype=float)
    offsets[:, 0] = (idx - (n_antennas - 1) / 2.0) * spacing_wavelengths
    return offsets


def steering_vector(
    element_offsets_wavelengths: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Far-field array steering vector for a given propagation direction.

    Parameters
    ----------
    element_offsets_wavelengths:
        Array of shape ``(n_antennas, 3)`` giving the position of each antenna
        element relative to the array centre, expressed in wavelengths.
    direction:
        Unit vector (length 3) pointing from the array towards the source.

    Returns
    -------
    numpy.ndarray
        Complex steering vector of shape ``(n_antennas,)`` with unit-modulus
        entries (so ``||a||**2 == n_antennas``).
    """
    phases = 2.0 * np.pi * (element_offsets_wavelengths @ direction)
    return np.exp(1j * phases)
