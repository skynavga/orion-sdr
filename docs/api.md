<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# API Reference

## Python API

All classes live in the flat `orion_sdr` namespace.

### Class Summary

#### Analog

| Class | Direction | Constructor |
| --- | --- | --- |
| `CwEnvelopeDemod` | IQ → audio | `(sample_rate, tone_hz, env_bw_hz)` |
| `AmEnvelopeDemod` | IQ → audio | `(fs, audio_bw_hz, abs_approx=False)` |
| `SsbProductDemod` | IQ → audio | `(fs, bfo_hz, audio_bw_hz)` |
| `FmQuadratureDemod` | IQ → audio | `(fs, dev_hz, audio_bw_hz)` |
| `PmQuadratureDemod` | IQ → audio | `(fs, k, audio_bw_hz)` |
| `AmDsbMod` | audio → IQ | `(fs, rf_hz, carrier_level, modulation_index)` |
| `CwKeyedMod` | audio → IQ | `(sample_rate, tone_hz, rise_ms, fall_ms)` |
| `FmPhaseAccumMod` | audio → IQ | `(sample_rate, deviation_hz, rf_hz)` |
| `PmDirectPhaseMod` | audio → IQ | `(sample_rate, kp_rad_per_unit, rf_hz)` |
| `SsbPhasingMod` | audio → IQ | `(fs, audio_bw_hz, audio_if_hz, rf_hz, usb)` |

#### Digital

| Class | Direction | Constructor | Notes |
| --- | --- | --- | --- |
| `BpskMod` | bits → IQ | `(fs, rf_hz, gain)` | 1 bit/symbol |
| `QpskMod` | bits → IQ | `(fs, rf_hz, gain)` | 2 bits/symbol; input length must be even |
| `QamMod` | bits → IQ | `(order, fs, rf_hz, gain)` | order ∈ {16, 64, 256}; 4/6/8 bits/symbol |
| `BpskDemod` | IQ → bits | `(gain)` | 1 bit/symbol out |
| `QpskDemod` | IQ → bits | `(gain)` | 2 bits/symbol out |
| `QamDemod` | IQ → bits | `(order, gain)` | order ∈ {16, 64, 256}; raises `ValueError` otherwise |

Digital classes fuse the mapper/decider and waveform stage into a single `process()` call.
Input bits are `uint8` arrays (one bit per byte, LSB used). Output IQ is `complex64`; output bits are `uint8`.

#### PSK31

| Class / function | Description |
| --- | --- |
| `VaricodeEncoder()` | `push_preamble(n)`, `push_byte(b)`, `push_postamble(n)`, `drain_bits()`, `is_empty()` |
| `VaricodeDecoder()` | `push_bits(bits: uint8[N])`, `pop_bytes() → bytes` |
| `Bpsk31Mod(fs, rf_hz, gain)` | `modulate_text(text, pre, post) → complex64[N]`, `modulate_bits(bits) → complex64[N]` |
| `Bpsk31Demod(fs, rf_hz, gain)` | `process(iq) → float32[N]` (one soft value per symbol) |
| `Bpsk31Decider()` | `process(soft) → uint8[N]` (threshold soft bits to hard decisions) |
| `Qpsk31Mod(fs, rf_hz, gain)` | Same API as `Bpsk31Mod` |
| `Qpsk31Demod(fs, rf_hz, gain)` | `process(iq) → float32[N]` (differential `[Re(d), Im(d)]` pairs) |
| `Psk31Stream(mode, fs, hz, gain)` | `feed(iq) → str`, `flush() → str`; mode = `"bpsk"` or `"qpsk"` |
| `psk31_sync(iq, fs, ...)` | Waterfall carrier search → `list[dict]` of candidates |
| `best_psk31_sync(candidates, hz)` | Pick best candidate from `psk31_sync()` results |

`psk31_sync` candidate dicts contain:
`{"time_sym": int, "freq_bin": int, "carrier_hz": float, "score": float, "soft_bits": float32[N]}`.
Use `Psk31Stream` for end-to-end decode, or manual pipeline via
`Bpsk31Demod` → `Bpsk31Decider` → `VaricodeDecoder`.

#### FT8 / FT4

| Class / function | Description |
| --- | --- |
| `Ft8Mod(fs, base_hz, rf_hz, gain)` | FT8 modulator: `modulate(tones: uint8[58]) → complex64[151680]` |
| `Ft8Demod(fs, base_hz)` | FT8 demodulator: `demodulate(iq: complex64[≥151680]) → uint8[58]` |
| `Ft8Codec` | Stateless; `encode(payload) → uint8[58]`, `decode_hard(tones) → bytes\|None`, `decode_soft(llr) → bytes\|None` |
| `Ft4Mod(fs, base_hz, rf_hz, gain)` | FT4 modulator: `modulate(tones: uint8[87]) → complex64[60480]` |
| `Ft4Demod(fs, base_hz)` | FT4 demodulator: `demodulate(iq: complex64[≥60480]) → uint8[87]` |
| `Ft4Codec` | Same interface as `Ft8Codec`; includes FT4 XOR scramble |
| `ft8_sync(iq, fs, base_hz, max_hz, t_min, t_max, max_cand)` | Returns `list[dict]` of sync candidates with soft LLRs |
| `ft4_sync(...)` | Same signature and return shape as `ft8_sync` |
| `ft8_pack_standard(call_to, call_de, extra)` | Pack standard QSO message → `bytes[10]` |
| `ft8_pack_free_text(text)` | Pack free-text message → `bytes[10]` |
| `ft8_pack_telemetry(data)` | Pack 9-byte telemetry blob → `bytes[10]` |
| `ft8_unpack(payload)` | Unpack `bytes[10]` → typed `dict` |

`ft8_sync` / `ft4_sync` candidate dicts contain:
`{"time_sym": int, "freq_bin": int, "score": float, "llr": float32[174]}`.
Pass `llr` to `Ft8Codec.decode_soft` / `Ft4Codec.decode_soft` to recover the payload.

`ft8_unpack` returns a dict whose `"type"` key is one of
`"standard"`, `"free_text"`, `"telemetry"`, `"nonstd"`, or `"unknown"`.

#### OFDM

| Class / function | Description |
| --- | --- |
| `OfdmConfig(n_fft, cp_len, data_carriers, pilot_idx, pilot_val, fs, rf_hz, gain, mode)` | Carrier plan + RF/constellation config |
| `OfdmMod(cfg)` | `modulate(bits) → complex64[]`; fused mapper+grid+IFFT+CP+upconversion |
| `OfdmDemod(cfg, equalizer="training_symbol")` | Fused CP-remove+FFT+equalize+grid-extract+decide |
| `OfdmRxFrame` | Getters: `bits`, `num_symbols`, `evm_db`, `cfo_hz`, `timing_offset_samples`, `channel_mse` |
| `build_ofdm_rx_frame(cfg, soft_symbols, bits)` | Builds an `OfdmRxFrame`; `evm_db` always populated |
| `ofdm_sync(iq, fs, num_repeats, repeat_len, search_start, search_end, ...)` | Schmidl & Cox preamble search → `list[dict]` |
| `generate_ofdm_preamble(cfg, num_repeats, repeat_len, ...)` | Generates the matching preamble IQ |

`OfdmConfig`'s `constellation` ∈ `"bpsk"|"qpsk"|"qam16"|"qam64"|"qam256"`;
raises `ValueError` on an unknown constellation or invalid carrier plan.
`OfdmDemod`'s `equalizer` ∈ `"training_symbol"|"pilot_interp"`; methods
`estimate_channel(training_iq)`, `demodulate(iq) → uint8[]`,
`demodulate_soft(iq) → (complex64[], uint8[])`. `OfdmRxFrame`'s
`cfo_hz`/`timing_offset_samples`/`channel_mse` are `None` until
acquisition/equalization has run. `ofdm_sync`/`generate_ofdm_preamble` take
optional `training_n_fft`/`training_cp_len` for wide-range integer-CFO recovery.

`ofdm_sync` candidate dicts contain:
`{"start_sample": int, "cfo_hz": float, "integer_cfo_bins": int, "score": float}`.
Total CFO is `cfo_hz + integer_cfo_bins * (fs / n_fft)`. `integer_cfo_bins` is
`0` unless `training_n_fft`/`training_cp_len` were supplied. Pass matching
`training_n_fft`/`training_cp_len` to both `generate_ofdm_preamble` and
`ofdm_sync` to enable wide-range integer-CFO recovery.

`CarrierGrid`/`FftBlock`/`GridMap` are not exposed to Python individually,
matching how the per-order symbol mappers aren't exposed today — `OfdmMod`
and `OfdmDemod` are the two main entry points.

### Array Types

| Domain | dtype | Notes |
| --- | --- | --- |
| IQ | `numpy.ndarray[complex64]` | 1-D, C-contiguous |
| Audio | `numpy.ndarray[float32]` | 1-D, C-contiguous |
| Tones | `numpy.ndarray[uint8]` | 1-D, values 0–7 (FT8) or 0–3 (FT4) |
| LLR | `numpy.ndarray[float32]` | 1-D, length 174; positive → bit likely 0 |

A wrong `dtype` or non-contiguous layout raises `ValueError`.

## Rust API

The Rust API is built around the `Block` trait from `src/core.rs`.
See [design.md](design.md) for the trait definition and [source.md](source.md)
for the full module layout.

### Graph Schedulers

| Type | Input | Output |
| --- | --- | --- |
| `IqToAudioChain` | `Complex32` | `f32` |
| `IqToIqChain` | `Complex32` | `Complex32` |
| `AudioToIqChain` | `f32` | `Complex32` |

OFDM's multi-stage, rate-changing pipeline (mapper → grid → IFFT/CP) is
deliberately **not** routed through these generic chains — see
[Multicarrier / OFDM Primitives](#multicarrier--ofdm-primitives) below.

### DSP Primitives

| Type | Description |
| --- | --- |
| `AgcRms` / `AgcRmsIq` | RMS-based automatic gain control |
| `DcBlocker` | 1st-order high-pass (y = x − x₁ + r·y₁) |
| `FirDecimator` | FIR anti-alias + integer decimation |
| `FirLowpass` | FIR low-pass filter |
| `Biquad` | Transposed Direct Form II biquad |
| `LpCascade` | Two cascaded biquads (4th-order) |
| `LpDcCascade` | Fused `LpCascade` + `DcBlocker` |
| `Nco` | Numerically controlled oscillator (phasor recurrence) |
| `Rotator` | Continuous phase rotator |

### Utilities

| Function / Constant | Description |
| --- | --- |
| `rms(x)` | Root-mean-square of a real slice |
| `tone(fs, f_hz, n, amp)` | Generate a real sine tone |
| `gen_complex_tone(fs, f_hz, n)` | Generate a complex baseband tone |
| `snr_db_at(fs, f_hz, x)` | Single-bin SNR using Hann window + DFT |
| `power_spectrum(samples, fs)` | Hann-windowed FFT power spectrum (dB); returns `(bins, bin_hz)` |
| `nb_spectrum_snr_db(samples, fs, carrier_hz)` | Narrowband: single peak bin vs. wideband noise-floor median |
| `wb_spectrum_snr_db(samples, fs, carrier_hz, occupied_hz)` | Wideband: mean in-band power vs. out-of-band median (for OFDM-like signals) |
| `spectrum_bw_hz(samples, fs, carrier_hz, threshold_db)` | Occupied bandwidth from power spectrum |
| `best_sync(results, carrier_hz, baud)` | Pick the best PSK31 sync result nearest to carrier |
| `atan2_approx(y, x)` | Fast 5th-order minimax atan2 approximation |
| `SIGNAL_THRESHOLD` | RMS threshold for silence detection (0.1) |
| `PSK31_BW_HZ` | PSK31 bandwidth: 2 × baud = 62.5 Hz |

### Analog Modulators / Demodulators

| Type | Description |
| --- | --- |
| `AmDsbMod` | Full-carrier AM (A3E) modulator |
| `CwKeyedMod` | CW keyed modulator with rise/fall shaping |
| `FmPhaseAccumMod` | Phase-accumulator FM modulator (phasor recurrence) |
| `PmDirectPhaseMod` | Direct-phase PM modulator |
| `SsbPhasingMod` | Weaver/phasing-method SSB modulator |
| `AmEnvelopeDemod` | AM envelope detector (PowerSqrt or AbsApprox) |
| `CwEnvelopeDemod` | CW tone envelope demodulator |
| `FmQuadratureDemod` | Quadrature FM discriminator (`atan2_approx`) |
| `PmQuadratureDemod` | Quadrature PM demodulator |
| `SsbProductDemod` | SSB product detector with BFO |

### Digital Modulators / Demodulators

| Type | Description |
| --- | --- |
| `BpskMapper` | u8 bits → C32 symbols (1 bit/symbol) |
| `BpskMod` | C32 symbols → C32 IQ (carrier upconversion) |
| `BpskDemod` | C32 IQ → C32 soft symbols |
| `BpskDecider` | C32 soft symbols → u8 bits |
| `QpskMapper` | u8 bits → C32 symbols (2 bits/symbol, Gray-coded, 1/√2 normalized) |
| `QpskMod` | C32 symbols → C32 IQ |
| `QpskDemod` | C32 IQ → C32 soft symbols |
| `QpskDecider` | C32 soft symbols → u8 bits (2 per symbol) |
| `QamMapper<BITS>` | u8 bits → C32 symbols; BITS ∈ {4,6,8} for 16/64/256-QAM |
| `QamMod` | C32 symbols → C32 IQ (order-independent) |
| `QamDemod` | C32 IQ → C32 soft symbols (order-independent) |
| `QamDecider<BITS>` | C32 soft symbols → u8 bits; BITS/2 I bits + BITS/2 Q bits per symbol |
| `Qam16Mapper` / `Qam16Decider` | Type aliases for `QamMapper<4>` / `QamDecider<4>` |
| `Qam64Mapper` / `Qam64Decider` | Type aliases for `QamMapper<6>` / `QamDecider<6>` |
| `Qam256Mapper` / `Qam256Decider` | Type aliases for `QamMapper<8>` / `QamDecider<8>` |

### PSK31 Waveform

| Type | Description |
| --- | --- |
| `Bpsk31Mod` | `modulate_text(text, preamble_bits, postamble_bits) → Vec<Complex32>`, `modulate_bits(bits) → Vec<Complex32>` |
| `Bpsk31Demod` | `process(iq, soft) → WorkReport` — one `f32` per symbol; positive = bit 1 (no phase flip) |
| `Bpsk31Decider` | `process(soft, bits) → WorkReport` — threshold at 0.0 |
| `Qpsk31Mod` | Same API as `Bpsk31Mod`; convolutional-encodes input bits before modulation |
| `Qpsk31Demod` | `process(iq, soft) → WorkReport` — two `f32` per symbol `[Re(d), Im(d)]` (differential detection product) |
| `Qpsk31Decider` | Buffers differential symbols; `flush(output)` runs Viterbi decode |

### PSK31 Codec

| Type | Description |
| --- | --- |
| `VaricodeEncoder` | `push_preamble(n)`, `push_byte(b)`, `push_postamble(n)`, `next_bit() → Option<u8>`, `drain_bits() → Vec<u8>` |
| `VaricodeDecoder` | `push_bit(bit)`, `pop_char() → Option<u8>` |
| `conv_encode(bits)` | Rate-1/2 K=5 encoder (G0=25, G1=23) → interleaved `Vec<u8>` |
| `viterbi_decode(soft)` | Soft Viterbi decoder; input `[Re(d), Im(d)]` DQPSK pairs; non-coherent branch metric |
| `viterbi_decode_coherent(soft, steps)` | Coherent MLSE Viterbi; tracks absolute phasor per state (retained for reference) |
| `viterbi_decode_hard(bits)` | Hard-decision wrapper around `viterbi_decode` |
| `StreamingViterbi` | Fixed-lag sliding-window Viterbi; `feed_symbol(re, im)`, `flush()` |
| `Psk31Stream` | Streaming decode; `new_bpsk(fs, hz, gain)` / `new_qpsk(...)`, `feed(iq)→String`, `flush()→String` |

### PSK31 Sync

| Function / Type | Description |
| --- | --- |
| `psk31_sync(iq, fs, base_hz, max_hz, ...)` | Waterfall energy-persistence carrier search → `Vec<Psk31SyncResult>` |
| `Psk31SyncResult` | `{time_sym: usize, freq_bin: usize, carrier_hz: f32, score: f32, soft_bits: Vec<f32>}` |

### FT8 / FT4 Waveform

| Type | Description |
| --- | --- |
| `Ft8Frame` | `[u8; 58]` — 58 Gray-coded tone indices (0–7) |
| `Ft8Mod` | Frame-at-a-time CPFSK modulator; `modulate(&Ft8Frame) → Vec<Complex32>` (151 680 samples) |
| `Ft8Demod` | Dot-product energy demodulator; `demodulate(&[Complex32]) → Option<Ft8Frame>` |
| `Ft4Frame` | `[u8; 87]` — 87 Gray-coded tone indices (0–3) |
| `Ft4Mod` | Same as `Ft8Mod`; output is 60 480 samples |
| `Ft4Demod` | Same as `Ft8Demod` for FT4 |

### FT8 / FT4 Codec

| Type | Description |
| --- | --- |
| `Ft8Codec` | `encode(&Ft8Bits) → Ft8Frame`, `decode_hard(&Ft8Frame) → Option<Ft8Bits>`, `decode_soft(&[f32; 174]) → Option<Ft8Bits>` |
| `Ft4Codec` | Same interface; includes FT4 XOR scramble before CRC+LDPC |
| `Ft8Bits` / `Ft4Bits` | `[u8; 10]` — 77-bit payload in 10 bytes (MSB-first, 3 slack bits) |

### FT8 / FT4 Sync

| Function / Type | Description |
| --- | --- |
| `ft8_sync(iq, fs, base_hz, max_hz, t_min, t_max, max_cand)` | Waterfall search → `Vec<Ft8SyncResult>` |
| `ft4_sync(...)` | Same for FT4 |
| `Ft8SyncResult` | `{time_sym: i32, freq_bin: usize, score: f32, llr: [f32; 174]}` |
| `Ft4SyncResult` | Same fields |

### FT8 / FT4 Message

| Function / Type | Description |
| --- | --- |
| `Ft8Message` | Enum: `Standard`, `FreeText`, `NonStd`, `Telemetry`, `Unknown` |
| `Payload77` | `[u8; 10]` — 77-bit payload |
| `GridField` | Enum: `Grid(String)`, `Report(i8)`, `RReport(i8)`, `RRR`, `RR73`, `Seventy3`, `None` |
| `pack77(msg, ht)` | `Ft8Message → Option<Payload77>` |
| `unpack77(payload, ht)` | `Payload77 → Ft8Message` |
| `CallsignHashTable` | In-memory hash table for nonstandard callsign resolution |

### Multicarrier / OFDM Primitives

Waveform-agnostic FFT-domain building blocks in `multicarrier/`, shared by
OFDM today and by planned future waveforms (DFT-s-OFDM/SC-FDMA, OTFS). See
[design.md](design.md#multicarrier--ofdm-pipeline) for the FFT-normalization,
carrier-indexing, and numerology conventions these types follow.

| Type | Description |
| --- | --- |
| `SubcarrierRole` | Enum: `Data`, `Pilot`, `Null` |
| `CarrierPlan` | Caller-owned resource-grid description: `n_fft`, `cp_len`, signed data/pilot carrier indices |
| `CarrierPlanError` | `OutOfRange(i32, usize)`, `Overlap(i32)`, `EmptyDataSet` |
| `CarrierGrid` | Resolved signed-index → rustfft-bin mapping for a `CarrierPlan`, built once via `from_plan` |
| `FftBlock` | Forward FFT (`C32 → C32`), unity gain, allocation-free (cached `rustfft` plan + scratch) |
| `IfftBlock` | Inverse FFT, `1/N` scale folded into the output copy |
| `CyclicPrefixInsert` | Prepends CP: `n_fft` in → `n_fft + cp_len` out |
| `CyclicPrefixRemove` | Removes CP: `n_fft + cp_len` in → `n_fft` out |
| `GridMap` | TX: dense data symbols → sparse `n_fft`-bin frequency vector (nulls zeroed, pilots inserted) |
| `GridExtract` | RX: `n_fft`-bin frequency vector → dense data-carrier stream (ignores pilots/channel) |

`CarrierPlan` builds via `with_data_carriers`/`with_pilot_carriers`, and
validates via `validate() -> Result<(), CarrierPlanError>`. `GridExtract`
deliberately ignores pilots/channel estimation — that's `OfdmEqualizer`'s
job, running upstream of it in the RX chain. Every type above is a
whole-symbol-per-call `Block`: a partial trailing chunk is a no-op
(`WorkReport::default()`), with no cross-call buffering.

### OFDM Waveform

| Type | Description |
| --- | --- |
| `ConstellationOrder` | Enum: `Bpsk\|Qpsk\|Qam16\|Qam64\|Qam256`; `bits_per_symbol()` |
| `OfdmConfig` | `carrier_plan`, `fs`, `rf_hz`, `gain`, `constellation`; `bits_per_ofdm_symbol()`, `samples_per_ofdm_symbol()` |
| `OfdmMod` | `Block<u8, C32>`: bits → mapper → `GridMap` → `IfftBlock` → `CyclicPrefixInsert` → optional `Rotator` |
| `OfdmDemod` | `Block<C32, C32>`: `CyclicPrefixRemove` → `FftBlock` → `GridExtract`, plus scalar gain correction |
| `OfdmDecider` | `Block<C32, u8>`: hard decision, dispatches by `ConstellationOrder` |
| `OfdmEqualizer` | `Block<C32, C32>`: channel equalizer, its own composable stage between `FftBlock` and `GridExtract` |
| `EqualizerMethod` | `TrainingSymbolHold` (default) \| `PerSymbolPilotInterp` (opt-in) |
| `OfdmSoftDemod` | `Block<C32, f32>`: soft (max-log LLR) demapper, dispatches by `ConstellationOrder` |
| `bpsk_soft_llr`/`qpsk_soft_llr`/`qam_soft_llr::<BITS>` | Per-order soft-LLR extraction; positive LLR ⇒ bit more likely 0 |
| `OfdmRxFrame` | Per-packet RX diagnostics: `bits`, `num_symbols`, `evm_db`, `cfo_hz`, `timing_offset_samples`, `channel_mse` |
| `build_ofdm_rx_frame(cfg, soft_symbols, bits)` | Builds an `OfdmRxFrame`; `evm_db` always populated |

`OfdmMod`'s mapper reuses `BpskMapper`/`QpskMapper`/`QamMapper<BITS>`
verbatim; its `modulate(bits) -> Vec<C32>` convenience wrapper zero-pads a
final partial symbol. `OfdmDemod` is the exact inverse of `OfdmMod`'s TX
chain. `OfdmEqualizer` is not fused into `OfdmDemod`, so it can be swapped or
disabled independently; `TrainingSymbolHold` estimates once per packet from
the training symbol and holds it, `PerSymbolPilotInterp` re-estimates every
symbol via pilot interpolation for time-varying/Doppler channels.
`OfdmSoftDemod` is a separate type from `OfdmDecider` — no mandatory FEC
ships with this crate, so LLRs are the deliverable. `OfdmRxFrame`'s
`Option`-typed fields (`cfo_hz`, `timing_offset_samples`, `channel_mse`) are
`None` until acquisition/equalization has actually run; `build_ofdm_rx_frame`
computes `evm_db` by re-mapping hard-decided bits to their ideal
constellation points and comparing to the soft symbols.

### OFDM Sync

| Type / Function | Description |
| --- | --- |
| `OfdmPreamble` | `num_repeats`, `repeat_len`, optional `training_symbol`; `with_training_symbol(n_fft, cp_len)` |
| `TrainingSymbol` | `n_fft`, `cp_len` — known symbol for integer-CFO recovery and channel estimation |
| `OfdmSyncResult` | `{start_sample, cfo_hz, integer_cfo_bins, score}` |
| `generate_ofdm_preamble(preamble, cfg) -> Vec<C32>` | Deterministic, reproducible preamble + training symbol |
| `ofdm_sync(iq, fs, preamble, search_start, search_end) -> Vec<OfdmSyncResult>` | Schmidl & Cox timing/CFO search |

`with_training_symbol` opts into wide-range integer-CFO recovery.
`OfdmSyncResult::cfo_hz` is fractional-only, unambiguous within
±`fs / (2·repeat_len)`; total CFO is
`cfo_hz + integer_cfo_bins as f32 * (fs / n_fft)`. `generate_ofdm_preamble`
uses a fixed-seed pseudo-random pattern, reproducible on both TX and RX
without shared state. `ofdm_sync` runs the integer-CFO search (when a
training symbol is present) on the top-5 timing candidates by score; results
are sorted by descending score.
