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
| `VaricodeEncoder()` | Stateful encoder; `push_preamble(n)`, `push_byte(b)`, `push_postamble(n)`, `drain_bits() → uint8[N]`, `is_empty() → bool` |
| `VaricodeDecoder()` | Stateful decoder; `push_bits(bits: uint8[N])`, `pop_bytes() → bytes` |
| `Bpsk31Mod(fs, rf_hz, gain)` | BPSK31 modulator; `modulate_text(text, preamble_bits, postamble_bits) → complex64[N]`, `modulate_bits(bits) → complex64[N]` |
| `Bpsk31Demod(fs, rf_hz, gain)` | BPSK31 demodulator; `process(iq) → float32[N]` (one soft value per symbol; positive = bit 1) |
| `Qpsk31Mod(fs, rf_hz, gain)` | QPSK31 modulator; same API as `Bpsk31Mod` |
| `Qpsk31Demod(fs, rf_hz, gain)` | QPSK31 demodulator; `process(iq) → float32[N]` (buffers phase-corrected phasor estimates), `flush() → uint8[N]` (runs coherent Viterbi MLSE) |
| `psk31_sync(iq, fs, base_hz, max_hz, ...)` | Waterfall carrier search → `list[dict]` of candidates with soft bits |

`psk31_sync` candidate dicts contain:
`{"time_sym": int, "freq_bin": int, "carrier_hz": float, "score": float, "soft_bits": float32[N]}`.
Pass `soft_bits` to `Bpsk31Demod` (BPSK31) or `Qpsk31Demod.flush()` (QPSK31) to decode.

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
| `BasicChain` | generic | generic |

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
| `Qpsk31Demod` | `process(iq, soft) → WorkReport` — two `f32` per symbol `[Re(sym_c), Im(sym_c)]` (phase-corrected absolute phasor) |
| `Qpsk31Decider` | Buffers phasor estimates; `flush(output)` runs coherent Viterbi MLSE |

### PSK31 Codec

| Type | Description |
| --- | --- |
| `VaricodeEncoder` | `push_preamble(n)`, `push_byte(b)`, `push_postamble(n)`, `next_bit() → Option<u8>`, `drain_bits() → Vec<u8>` |
| `VaricodeDecoder` | `push_bit(bit)`, `pop_char() → Option<u8>` |
| `conv_encode(bits)` | Rate-1/2 K=5 encoder (G0=25, G1=23) → interleaved `Vec<u8>` |
| `viterbi_decode(soft)` | Differential soft Viterbi decoder; input `[Re(d), Im(d)]` pairs, positive = coded bit likely 0 |
| `viterbi_decode_coherent(soft, phase_steps)` | Coherent MLSE Viterbi; input `[Re(sym_c), Im(sym_c)]` pairs; tracks hypothesised absolute phasor per trellis state |
| `viterbi_decode_hard(bits)` | Hard-decision wrapper around `viterbi_decode` |

### PSK31 Sync

| Function / Type | Description |
| --- | --- |
| `psk31_sync(iq, fs, base_hz, max_hz, min_carrier_syms, peak_margin_db, n_bits, max_cand)` | Waterfall energy-persistence carrier search → `Vec<Psk31SyncResult>` |
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
