# AirComp — Over-the-Air Computation Simulation Framework

A modular Python framework for simulating **analog over-the-air computation
(AirComp)** in an **indoor** wireless environment, where a *small* number of
**hosts** (access points / parameter servers) aggregate signals transmitted
concurrently by a *large* number of **agents** (edge devices / sensors). Both
hosts and agents can be single- or multi-antenna (fully configurable).

AirComp exploits the signal-superposition property of the wireless
multiple-access channel: when all agents transmit at once, the host receives the
channel-weighted **sum** of their signals. With appropriate pre-/post-processing
this realises a *nomographic* function

```
f(s_1, …, s_K) = ψ( Σ_k φ_k(s_k) )
```

such as the arithmetic mean (the canonical aggregation step in federated
learning), a weighted sum, or the geometric mean — in a **single channel use**,
regardless of the number of agents `K`.

## Why this regime

In realistic indoor deployments there are far more devices than access points
(`n_hosts ≪ n_agents`). The framework models this directly: each host carries an
antenna array, and a central processor performs **joint aggregation beamforming**
across all cooperating hosts (a cell-free-style setup), so the few hosts still
provide enough receive dimensions to serve many agents.

## Signal model (MIMO)

Antenna counts on **both** ends are configurable: agents have `N_t` transmit
antennas and hosts have `N_r` receive antennas each (either can be 1). `K` agents
transmit standardised symbols `x` (with `E|x_k|² = 1`) through transmit
beamforming vectors `b_k ∈ ℂ^{N_t}` with power budget `‖b_k‖² ≤ P_k`. With the
stacked MIMO channel `H ∈ ℂ^{R×K×N_t}` (`R = n_hosts × N_r`), the received signal is

```
y = Σ_k H_k b_k x_k + n,        n ~ CN(0, σ² I)
```

The host applies an aggregation combiner `m ∈ ℂ^R` and a denoising factor `η` to
estimate the target sum `T = Σ_k x_k`:

```
T̂ = (mᴴ y) / √η
```

For a fixed `m`, the effective per-agent channel is `f_k = H_kᴴ m ∈ ℂ^{N_t}`. Each
agent uses **maximum-ratio transmission** `b_k = √η · f_k / ‖f_k‖²` to hit the
target gain with least power, so **channel-inversion power control** gives

```
η = min_k P_k ‖H_kᴴ m‖²           (the worst agent limits the whole computation)
MSE = ‖m‖² σ² / η
```

For single-antenna agents (`N_t = 1`) this reduces exactly to the scalar case
`‖H_kᴴ m‖² = |mᴴ h_k|²`; for single-antenna hosts (`R = 1`) the combiner is a
scalar and MRC coincides with the optimised design.

### Channel model (`aircomp/channel.py`)

Each host–agent link is a matrix `H_k = √β · H_small`:

* **Large-scale gain `β`** — log-distance path loss with a free-space intercept,
  configurable path-loss exponent (indoor ≈ 1.6–3.5), plus log-normal shadowing.
* **Small-scale `H_small`** — Rician fading (configurable `K`-factor) whose LoS
  part is the rank-one outer product `a_rx a_txᴴ` of **geometry-derived steering
  vectors** for the host and agent ULAs, plus an i.i.d. Rayleigh NLoS matrix.
* **Noise** — thermal `kTB` over the configured bandwidth and receiver noise figure.

### Receivers (`aircomp/aggregation.py`)

| Aggregator | Combiner design |
|---|---|
| `ChannelInversionAggregator("mrc")` | Maximum-ratio combining, `m = Σ_{k,t} H[:,k,t]` (baseline) |
| `ChannelInversionAggregator(m)` | Any user-supplied fixed combiner |
| `OptimizedBeamformingAggregator` | Maximises the worst-agent gain `min_k ‖H_kᴴm‖²/‖m‖²` via Riemannian (unit-sphere) gradient ascent on a smooth soft-min surrogate — directly minimising the channel-inversion MSE |

Both aggregators assume MRT transmit beamforming at multi-antenna agents; the
transmit and receive designs together form the alternating-optimal joint scheme.

## Supported operations

Every operation reduces to the AirComp sum primitive `ψ(Σ_k φ_k(s_k))`; see
[functions.py](aircomp/functions.py).

| Family | Operation | φ_k (pre) / ψ (post) |
|---|---|---|
| **Natural-medium** | Addition | Σ s_k (identity) |
| | Subtraction | Σ ε_k s_k, `ε_k ∈ {±1}` |
| **Nomographic** | Arithmetic mean | ψ = ·/K |
| | Weighted average | φ = w_k s_k, ψ = ·/Σw_k |
| | Geometric mean | φ = ln s_k, ψ = exp(·/K) |
| | Euclidean norm | φ = s_k², ψ = √· |
| | Polynomial sum | φ = p(s_k) |
| **Advanced / approx.** | Majority vote | φ = 2v_k−1, ψ = sign |
| | Counting | φ = 1[s_k>τ], ψ = round |
| | Histogram | φ = one-hot(bin), ψ = round (D sums) |
| | Maximum | φ = s_k^p, ψ = ·^{1/p} |
| | Minimum | φ = s_k^{−p}, ψ = ·^{−1/p} |

### Principle and what each operation depends on

AirComp is **analog**: every agent standardises its pre-processed value `φ_k(s_k)`
to unit power, a DAC turns it into a baseband symbol, transmit precoding +
channel-inversion power control align the contributions, and the host's antennas
superimpose them into `Σ_k φ_k(s_k)` — which an ADC samples and `ψ` turns into the
result. Three things therefore set the accuracy of *every* operation:

* **Channel condition** — the denoising factor `η = min_k P_k‖H_kᴴm‖²` is fixed by
  the *worst* agent's effective gain, and the computation error is `MSE = ‖m‖²σ²/η`.
  A deep fade or heavy shadowing on any single agent (a "straggler") throttles the
  whole computation, so accuracy tracks the tail of the channel distribution, not
  the average.
* **Antennas** — more **host** antennas give receive-beamforming gain and let `m`
  steer to lift the worst agent (raising `η`); more **agent** antennas give
  transmit (MRT) array gain. Both raise the computation SNR and lower NMSE.
* **DAC / ADC (amplitude *and* sampling resolution)** — two orthogonal converter
  limits. *Amplitude*: finite bits plus clipping set a quantisation floor, and
  pre-processing that spreads the `φ_k` over a wide dynamic range raises PAPR and
  eats converter headroom. *Sampling*: oversampling (samples per symbol) spreads
  quantisation noise over a wider band, so matched-filtering back to one symbol
  recovers ~`10·log10(OSR)` dB of it — trading sample rate for effective bits.

#### Tuning options

All three dependencies are directly tunable from `SystemConfig`.

**Channel condition** — raise the operating SNR or ease the propagation:

| Knob | Field | Effect |
|---|---|---|
| Transmit power | `SystemConfig.tx_power_dbm` | +1 dBm ≈ −1 dB NMSE (until converter/approx-limited) |
| Path-loss exponent | `channel.path_loss_exponent` | lower (≈1.8) → stronger links; raise for harsher indoor |
| Shadowing spread | `channel.shadowing_std_db` | smaller → fewer deep-fade stragglers → lower NMSE |
| LoS strength | `channel.rician_k_db` | higher K → stronger LoS, so steering/beamforming bites harder |
| Noise | `channel.bandwidth_hz`, `channel.noise_figure_db` | narrower band / lower NF → lower noise power `σ²` |
| Receiver | `aggregator` | `OptimizedBeamformingAggregator` raises the worst-agent gain vs. MRC |
| Geometry | `room.*`, `n_hosts` placement | smaller room / more hosts → shorter links |

**Antennas** — add receive/transmit dimensions:

| Knob | Field | Effect |
|---|---|---|
| Host antennas | `SystemConfig.host_antennas` | receive-beamforming + diversity gain (monotonic) |
| Number of hosts | `SystemConfig.n_hosts` | more joint receive dimensions (cell-free aggregation) |
| Agent antennas | `SystemConfig.agent_antennas` | transmit-MRT array gain (≈ +3 dB per doubling) |
| Array spacing | `SystemConfig.array_spacing_wavelengths` | wider spacing → less correlated fading |
| Solver effort | `OptimizedBeamformingAggregator(n_iter, step_size, smoothing)` | more iterations → closer to the max-min optimum |

**DAC / ADC** — now modelled via `SystemConfig.converters` (`ConverterConfig`,
ideal by default). Quantisation costs roughly **6 dB NMSE per bit**:

| Knob | Field | Effect |
|---|---|---|
| DAC amplitude resolution | `converters.dac_bits` | `None` = ideal; finite bits add transmit quantisation (~6 dB/bit) |
| ADC amplitude resolution | `converters.adc_bits` | `None` = ideal; finite bits add receive quantisation (~6 dB/bit) |
| DAC headroom | `converters.dac_clip_sigma` | full-scale peak / RMS; small → clips peaks, large → wastes levels |
| ADC headroom | `converters.adc_clip_sigma` | same trade-off at the receiver AGC |
| DAC sampling resolution | `converters.dac_oversampling` | OSR (samples/symbol); +~3 dB per doubling (~0.5 bit/octave) |
| ADC sampling resolution | `converters.adc_oversampling` | OSR at the receiver; same processing gain |

Amplitude bits and oversampling combine: e.g. a 6-bit ADC at `OSR = 16` recovers
about `10·log10(16) ≈ 12 dB`, roughly a 8-bit converter at Nyquist.

```python
from aircomp import SystemConfig, ConverterConfig
cfg = SystemConfig(host_antennas=16, agent_antennas=4, tx_power_dbm=25,
                   converters=ConverterConfig(dac_bits=10, adc_bits=10,
                                              dac_oversampling=4, adc_oversampling=4))
```

Per-operation principle and sensitivities:

* **Addition** — the medium sums the transmitted amplitudes directly; accuracy is
  the raw computation SNR. Balanced dynamic range, so easy on the DAC/ADC.
* **Subtraction** — identical to addition with a `±1` sign folded into each symbol;
  same channel/antenna/converter behaviour.
* **Arithmetic mean** — sum then scale by `1/K`; the `1/K` also shrinks the noise,
  so the mean is *more* robust than the raw sum as the agent count grows.
* **Weighted average** — agents pre-scale by `w_k`; a wide weight spread widens the
  DAC dynamic range and makes the low-weight agents' channels matter less.
* **Geometric mean** — sum in the log domain, then `exp`; the log **compresses**
  dynamic range, making this the most converter- and noise-friendly operation
  (best NMSE in the benchmark), but it needs strictly positive data.
* **Euclidean norm** — square, sum, `√`; squaring widens dynamic range (the largest
  agents dominate), while the final `√` compresses the residual error.
* **Polynomial sum** — each agent evaluates the polynomial locally, then sums;
  high-degree terms expand the dynamic range (DAC/ADC headroom) and amplify each
  agent's noise sensitivity.
* **Majority vote** — sum of `±1` votes, decided by sign; robust to noise except
  near a tie (small `|Σ|`), where a fade can flip the outcome. Accuracy is set by
  the vote margin relative to the computation SNR, not by raw NMSE — hence the
  exact-match rate is the meaningful metric.
* **Counting** — sum of indicators, then round; the integer count is recovered
  *exactly* whenever the SNR keeps the sum error below ½, so more antennas / power
  push it to 100% exact.
* **Histogram** — one counting sum per bin, so it costs `D` channel uses (sharing
  the same beamformer); low-count bins are the most noise-sensitive.
* **Maximum** — the `p`-norm surrogate `(Σ s_k^p)^{1/p}`: larger `p` sharpens the
  approximation but `s_k^p` blows up the DAC dynamic range and ADC quantisation, so
  `p` trades approximation error against converter/noise error. In the benchmark the
  approximation dominates and the channel is almost irrelevant.
* **Minimum** — the same surrogate with a negative exponent; small values raised to
  `−p` create the widest dynamic range of all the operations, making it the most
  demanding on converter resolution.

Run the accuracy benchmark for all of them:

```bash
python examples/evaluate_operations.py
```

It reports, per operation, the NMSE vs. the exact value, RMSE/MAE, the noiseless
**approximation floor** (for the approximated ops), and the **exact-match rate**
(for the discrete ops). At a healthy SNR the exact ops reach −90 dB or better,
counting/histogram recover exactly, majority vote matches ~94%, and max/min are
limited by their p-norm approximation rather than the channel.

```python
from aircomp import evaluate_operation, MaxApprox, SystemConfig
res = evaluate_operation(MaxApprox(p=6), SystemConfig(n_agents=40), n_trials=300)
print(res.summary())
```

## Operation principle diagrams

**Signal model.** Every operation reduces to one use of the classic linear model

```
y = H s + n
```

* `s = (s₁ … s_K)ᵀ` — transmit symbols, `sₖ = standardise(φ(dₖ))` with `E|sₖ|² = 1`.
* `H = (h₁ … h_K) ∈ ℂ^{R×K}` — effective channel, column `hₖ = Hₖ bₖ` is the indoor
  MIMO channel `Hₖ` (path loss · Rician fading) seen through the agent's MRT
  precoder `bₖ` (`‖bₖ‖² ≤ Pₖ`); `R = n_hosts × N_r` receive antennas.
* `n ~ CN(0, σ²I)` — receiver AWGN (`σ²` from bandwidth + noise figure).

The host recovers the aggregate with combiner `m` and denoising `η`:

```
ŝ_Σ = mᴴ y / √η  =  Σₖ (mᴴhₖ / √η) sₖ  +  mᴴn / √η  →  Σₖ sₖ + noise
```

channel-inversion / MRT sets `mᴴhₖ = √η` for every `k` (with
`η = minₖ Pₖ‖Hₖᴴm‖²`), so the wanted sum survives and only `mᴴn/√η` is left.
De-standardising and applying `ψ` yields the result:

```mermaid
flowchart LR
    subgraph TX["Agents k = 1..K  ·  N_t antennas"]
      d["data dₖ"] -->|"φ"| g["gₖ = φ(dₖ)"]
      g -->|"standardise"| sk["sₖ = (gₖ − μ)/σ"]
      sk -->|"DAC · MRT bₖ"| xk["send bₖ sₖ , ‖bₖ‖² ≤ Pₖ"]
    end
    xk --> mdl((("y = H s + n<br/>hₖ = Hₖ bₖ (fading · path loss)<br/>n ~ CN(0, σ²I)")))
    mdl --> rx
    subgraph RX["Hosts · n_hosts × N_r antennas"]
      rx["combine · ADC<br/>ŝ_Σ = mᴴy / √η<br/>η = minₖ Pₖ‖Hₖᴴm‖²"] -->|"de-standardise"| sig["Σₖ gₖ = σ·ŝ_Σ + Kμ"]
      sig -->|"ψ"| out["f(d₁ … d_K)"]
    end
```

### Symbol reference

Every symbol used in the equations and diagrams:

**Data and processing**

| Symbol | Meaning |
|---|---|
| `dₖ` | raw data value contributed by agent `k` |
| `K` | number of agents |
| `φ(·)` | per-agent **pre-processing** (operation-specific), applied before transmit |
| `gₖ = φ(dₖ)` | pre-processed value at agent `k` |
| `μ, σ` | mean and standard deviation of `{gₖ}` across agents (known at the host) |
| `sₖ = (gₖ − μ)/σ` | **standardised transmit symbol**, unit power `E\|sₖ\|² = 1` |
| `ψ(·)` | host **post-processing** (operation-specific), applied to the recovered sum |
| `f(d₁ … d_K)` | the operation's output value |

**Antennas, precoding, converters**

| Symbol | Meaning |
|---|---|
| `N_t` | transmit antennas per agent |
| `N_r` | receive antennas per host |
| `n_hosts` | number of hosts |
| `R = n_hosts × N_r` | total receive antennas (joint aggregation dimension) |
| `bₖ ∈ ℂ^{N_t}` | agent `k` transmit beamformer (**MRT** precoder) |
| `Pₖ` | transmit power budget of agent `k`, constraint `‖bₖ‖² ≤ Pₖ` |
| `MRT` | maximum-ratio transmission (transmit beamforming) |
| `DAC` / `ADC` | digital-to-analog / analog-to-digital converter (bits + oversampling) |

**Channel and received signal** (`y = H s + n`)

| Symbol | Meaning |
|---|---|
| `Hₖ ∈ ℂ^{R×N_t}` | MIMO channel from agent `k` to all receive antennas (path loss · Rician fading) |
| `hₖ = Hₖ bₖ ∈ ℂ^{R}` | **effective channel** column (physical channel through the precoder) |
| `H = (h₁ … h_K) ∈ ℂ^{R×K}` | stacked effective channel matrix |
| `s = (s₁ … s_K)ᵀ` | transmit-symbol vector |
| `n ~ CN(0, σ²I)` | receiver noise vector (circularly-symmetric complex Gaussian) |
| `σ²` | noise power per receive antenna (from bandwidth + noise figure) |
| `y ∈ ℂ^{R}` | stacked received signal, `y = H s + n` |

**Aggregation and recovery**

| Symbol | Meaning |
|---|---|
| `m ∈ ℂ^{R}` | receive **aggregation combiner** (beamformer) |
| `(·)ᴴ` | conjugate transpose (Hermitian); `mᴴhₖ` = combined effective gain of agent `k` |
| `‖·‖` | Euclidean norm |
| `η = minₖ Pₖ‖Hₖᴴm‖²` | **denoising factor**, set by the worst agent |
| `ŝ_Σ = mᴴy / √η` | estimate of the wanted sum `Σₖ sₖ` |
| `mᴴn / √η` | residual noise on the estimate |
| `Σₖ gₖ = σ·ŝ_Σ + Kμ` | de-standardised aggregate, before `ψ` |

**Operation-specific parameters** (per-operation diagrams)

| Symbol | Meaning |
|---|---|
| `εₖ ∈ {+1, −1}` | per-agent sign (subtraction) |
| `wₖ` | per-agent weight (weighted sum / average) |
| `τ` | threshold (counting) |
| `vₖ ∈ {0, 1}` | binary vote (majority) |
| `ind(·)` | indicator: 1 if the condition holds, else 0 |
| `one-hot(bin)` | length-`D` indicator selecting `dₖ`'s histogram bin (`D` = #bins) |
| `p` | order of the p-norm surrogate (max / min approximation) |
| `Σₖ`, `Πₖ` | sum / product over agents `k = 1..K` |

Each per-operation diagram below instantiates the `y = H s + n` pipeline —
filling in `φ` (what forms `sₖ`) and `ψ` (what the host does with `Σₖ sₖ`) — and
**highlights the stage the operation stresses**: red = limiting/most-demanding,
green = a feature it is easy on.

### Natural-medium

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = dₖ"| tx["sₖ = std(dₖ)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>residual = mᴴn/√η · limited by η"]
    rx -->|"ψ = identity"| r["Σₖ dₖ  ·  addition"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class rx hot;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = εₖ dₖ , εₖ = ±1"| tx["sₖ = std(εₖ dₖ)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>residual = mᴴn/√η · limited by η"]
    rx -->|"ψ = identity"| r["Σₖ εₖ dₖ  ·  subtraction"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class rx hot;
```

### Nomographic (pre-/post-processed)

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = dₖ"| tx["sₖ = std(dₖ)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>ψ = ÷K scales noise mᴴn/√η by 1/K"]
    rx -->|"ψ = ·/K"| r["(1/K) Σₖ dₖ  ·  arithmetic mean"]
    classDef cool fill:#e9f6ea,stroke:#59A14F,color:#143;
    class rx cool;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = wₖ dₖ"| tx["sₖ = std(wₖ dₖ) · DAC · MRT<br/>weight spread widens DAC range"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>η = minₖ Pₖ‖Hₖᴴm‖²"]
    rx -->|"ψ = ·/Σwₖ"| r["Σ wₖdₖ / Σ wₖ  ·  weighted average"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class tx hot;
```

```mermaid
flowchart LR
    d["dₖ > 0"] -->|"φ = ln dₖ"| tx["sₖ = std(ln dₖ) · DAC · MRT<br/>log compresses range → few bits OK"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>η = minₖ Pₖ‖Hₖᴴm‖²"]
    rx -->|"ψ = exp(·/K)"| r["(Πₖ dₖ) ^ (1/K)  ·  geometric mean"]
    classDef cool fill:#e9f6ea,stroke:#59A14F,color:#143;
    class tx cool;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = dₖ²"| tx["sₖ = std(dₖ²) · DAC · MRT<br/>square widens range (largest agents)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>ψ = √· compresses residual error"]
    rx -->|"ψ = √·"| r["√(Σₖ dₖ²)  ·  Euclidean norm"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class tx hot;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = p(dₖ)"| tx["sₖ = std(p(dₖ)) · DAC · MRT<br/>high-degree terms widen DAC/ADC range"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>η = minₖ Pₖ‖Hₖᴴm‖²"]
    rx -->|"ψ = identity"| r["Σₖ p(dₖ)  ·  polynomial sum"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class tx hot;
```

### Advanced / approximated

```mermaid
flowchart LR
    d["vote vₖ = 0/1"] -->|"φ = 2vₖ − 1"| tx["sₖ = std(2vₖ−1)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>sign flips when |mᴴn/√η| > margin"]
    rx -->|"ψ = (· > 0)"| r["majority bit  ·  flips near ties"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class rx hot;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = ind(dₖ > τ)"| tx["sₖ = std(ind)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>round exact if |mᴴn/√η| < ½"]
    rx -->|"ψ = round(·)"| r["count(dₖ > τ)  ·  counting"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class rx hot;
```

```mermaid
flowchart LR
    d["dₖ"] -->|"φ = one-hot(bin)"| tx["sₖ = std(one-hot)<br/>DAC · MRT bₖ (N_t ant.)"]
    tx --> y((("y = H s + n  ·  per bin<br/>D channel uses, shared m<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η per bin (N_r ant.)<br/>low-count bins are noise-sensitive"]
    rx -->|"ψ = round(·)"| r["bin counts  ·  histogram"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class y hot;
```

```mermaid
flowchart LR
    d["dₖ > 0"] -->|"φ = dₖ ^ p"| tx["sₖ = std(dₖ^p) · DAC · MRT<br/>dₖ^p → high PAPR → needs bits/headroom"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>p sets the approximation error"]
    rx -->|"ψ = · ^ (1/p)"| r["≈ maxₖ dₖ  ·  p-norm"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class tx hot;
```

```mermaid
flowchart LR
    d["dₖ > 0"] -->|"φ = dₖ ^ (−p)"| tx["sₖ = std(dₖ^−p) · DAC · MRT<br/>widest dynamic range → most bits needed"]
    tx --> y((("y = H s + n<br/>hₖ = Hₖbₖ · fading + path loss<br/>n ~ CN(0, σ²I)")))
    y --> rx["ŝ_Σ = mᴴy/√η (N_r ant.)<br/>p sets the approximation error"]
    rx -->|"ψ = · ^ (−1/p)"| r["≈ minₖ dₖ  ·  p-norm"]
    classDef hot fill:#ffe8e6,stroke:#E15759,color:#611;
    class tx hot;
```

## Package layout

```
aircomp/
├── __init__.py       # public API
├── config.py         # RoomConfig, ChannelConfig, SystemConfig dataclasses
├── utils.py          # dB/linear + dBm/W conversions, RNG, array steering
├── nodes.py          # Agent and Host (ULA antenna geometry)
├── environment.py    # indoor room + random agent / ceiling host placement
├── channel.py        # indoor path-loss + Rician fading MIMO channel generator
├── functions.py      # operation library (natural-medium / nomographic / advanced)
├── aggregation.py    # AirComp receivers: beamforming + power control
├── metrics.py        # MSE / NMSE (linear and dB)
├── evaluation.py     # per-operation accuracy benchmarking
└── simulator.py      # Monte-Carlo orchestration + parameter sweeps
examples/
├── quickstart.py             # one scenario, one Monte-Carlo run
├── run_demo.py               # NMSE vs {tx power, #agents, #antennas}
└── evaluate_operations.py    # accuracy of every supported operation
```

## Installation

```bash
pip install -r requirements.txt
```

Only `numpy` is required for the core library; `matplotlib` is used by the
example plots.

## Quickstart

```python
from aircomp import Simulator, SystemConfig, OptimizedBeamformingAggregator, ArithmeticMean

config = SystemConfig(n_agents=50, n_hosts=1, host_antennas=8, agent_antennas=2,
                      tx_power_dbm=10.0, seed=0)
sim = Simulator(config,
                aggregator=OptimizedBeamformingAggregator(n_iter=200),
                function=ArithmeticMean())

result = sim.run_monte_carlo(n_trials=500)
print(f"Computation NMSE: {result.nmse_db:.2f} dB")
```

Parameter sweeps:

```python
import numpy as np
sim.sweep_tx_power(np.arange(-10, 31, 5))   # NMSE vs SNR
sim.sweep_n_agents([5, 10, 20, 40, 80, 160])
sim.sweep_host_antennas([1, 2, 4, 8, 16, 32])   # receive-beamforming gain
sim.sweep_agent_antennas([1, 2, 4, 8])           # transmit-beamforming gain
```

Run the full demo (writes PNGs into `examples/`):

```bash
python examples/run_demo.py
```

## Representative results

Scenario: 40 agents, 2 hosts × 8 antennas, 2.4 GHz indoor office.

* **vs transmit power** — MSE ∝ 1/P for both receivers; the optimised
  beamformer gives a steady ≈ 15 dB NMSE gain over MRC.
* **vs number of agents** — MRC degrades as the worst-agent bottleneck bites,
  while the optimised beamformer stays essentially flat, sustaining accuracy as
  the crowd grows.
* **vs host antennas** — the optimised beamformer's NMSE improves monotonically,
  reflecting the aggregation-diversity gain of more receive dimensions.

## Extending the framework

* **New operations** — subclass `Operation` (`functions.py`) and implement
  `pre`/`post` (plus `sample_data`/`true_value` to benchmark it); add it to the
  list passed to `evaluate_operations`.
* **New receivers** — subclass `Aggregator` (`aggregation.py`) and implement
  `design`; reuse `_channel_inversion` for the power-control step.
* **New channel effects** — extend `IndoorChannel.realize` (e.g. spatial
  correlation, blockage, mobility).
* **New topologies** — pass explicit `agents=`/`hosts=` lists to
  `IndoorEnvironment`, or override the placement methods.

## Background reading

The models follow the mainstream AirComp literature on analog function
computation over multiple-access channels, uniform-forcing / channel-inversion
transceiver design, and receive-beamforming (aggregation) MSE minimisation for
multi-antenna AirComp and federated edge learning.
