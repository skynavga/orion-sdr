<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Acronym Glossary

Expansions for the acronyms used across the `orion-sdr` source and docs. See
[design.md](design.md) for crate-wide design patterns and [ofdm.md](ofdm.md) for
the OFDM/COFDM conventions.

| Acronym | Expansion | Notes |
| ------- | --------- | ----- |
| AFC | Automatic Frequency Control | PSK31's symbol-rate decision-directed PLL, tracks residual carrier drift |
| AGC | Automatic Gain Control | `AgcRms`, `AgcRmsIq` in `dsp/agc.rs` |
| AM | Amplitude Modulation | DSB (double-sideband) variant implemented |
| AWGN | Additive White Gaussian Noise | Standard noise model used in tests |
| BCH | Bose–Chaudhuri–Hocquenghem | Binary block code (`fec/bch.rs`); COFDM outer FEC option |
| BFO | Beat Frequency Oscillator | `bfo_hz` parameter in `SsbProductDemod` |
| BM | Berlekamp–Massey | Error-locator algorithm in the BCH/RS decoders |
| BP | Belief Propagation | Iterative sum-product algorithm used in LDPC decoders |
| BPSK | Binary Phase-Shift Keying | 1 bit/symbol; the COFDM header's fixed modulation |
| CFO | Carrier Frequency Offset | TX/RX oscillator mismatch; corrected by `Rotator` before OFDM demod |
| CLT | Central Limit Theorem | Used in AWGN generation (sum-of-uniforms approximation) |
| COFDM | Coded OFDM | The concatenated-FEC, framed OFDM link; see [ofdm.md](ofdm.md) |
| CP | Cyclic Prefix | `CyclicPrefixInsert`/`CyclicPrefixRemove`; absorbs multipath delay spread up to `cp_len` |
| CPFSK | Continuous-Phase Frequency-Shift Keying | Phase continuity across symbol boundaries; used by FT8/FT4 |
| CRC | Cyclic Redundancy Check | CRC-14 (0x2757) for FT8/FT4; generic CRC-16/CRC-32 (`fec`) for COFDM frames |
| CSR | Compressed Sparse Row | Flat contiguous edge-message storage (buffer + offset table) in the LDPC decoder (`fec/ldpc_codes.rs`) |
| CW | Continuous Wave | Morse-code keyed carrier |
| DATV | Digital Amateur Television | Amateur digital TV; NB-DVB-T is the DVB-T-over-ham-bands variant |
| DBPSK | Differential Binary Phase-Shift Keying | PSK31's BPSK31; also DVB-T TPS signalling (differential along the symbol axis) |
| DC | Direct Current | Zero-frequency component; blocked by `DcBlocker`; implicitly null in OFDM carrier plans |
| DQPSK | Differential Quadrature Phase-Shift Keying | PSK31's QPSK31 modulation, decoded via soft Viterbi |
| DSB | Double-Sideband | Both sidebands transmitted; see AM |
| DSP | Digital Signal Processing | — |
| DVB-H | Digital Video Broadcasting – Handheld | Mobile DVB variant; reuses DVB-T's TPS signalling verbatim (extra bits) |
| DVB-T | Digital Video Broadcasting – Terrestrial | Terrestrial digital-TV standard (ETSI EN 300 744); RS(204,188)+conv FEC, OFDM 2K/8K |
| DVB-T2 | Digital Video Broadcasting – Terrestrial 2nd gen | Successor to DVB-T; replaces TPS with P1/P2 L1 signalling (context only) |
| EHF | Extremely High Frequency | 30–300 GHz; upper end of the OFDM target-band range |
| EVM | Error Vector Magnitude | Soft-vs-ideal constellation distance, in dB; `OfdmRxFrame::evm_db` |
| FEC | Forward Error Correction | LDPC in FT8/FT4; COFDM concatenates an inner (LDPC/convolutional) and outer (BCH/RS) code (`fec/`) |
| FFT | Fast Fourier Transform | `FftBlock` (unity-gain forward) in `multicarrier/fft.rs`; the OFDM demod's frequency-domain transform |
| FIR | Finite Impulse Response | `FirLowpass`, `FirDecimator` in `dsp/` |
| FM | Frequency Modulation | Quadrature (discriminator) demod |
| FMA | Fused Multiply-Add | `f32::mul_add`; used throughout inner loops |
| FSK | Frequency-Shift Keying | Base modulation for FT8 (8-FSK) and FT4 (4-FSK) |
| FT4 | Fast Telegraphy 4-FSK | 4-FSK weak-signal mode; 6-second transmit period |
| FT8 | Fast Telegraphy 8-FSK | 8-FSK weak-signal mode; 15-second transmit period |
| GF | Galois Field | `Gf256` = GF(2^8) arithmetic (`fec/gf.rs`), foundation for BCH/RS; DVB-T TPS BCH uses GF(2^7) |
| GI | Guard Interval | The OFDM cyclic prefix as a fraction of the useful symbol (DVB-T: 1/32, 1/16, 1/8, 1/4); `cp_len = n_fft · GI` |
| HF | High Frequency | 3–30 MHz; primary target band for FT8/FT4 |
| IF | Intermediate Frequency | `rf_hz` parameter in modulators |
| IFFT | Inverse Fast Fourier Transform | `IfftBlock` (1/N-scaled) in `multicarrier/fft.rs`; the OFDM modulator's time-domain synthesis |
| IIR | Infinite Impulse Response | `Biquad`, `LpCascade` in `dsp/` |
| IQ | In-phase / Quadrature | Complex baseband representation; `Complex32` throughout |
| ISI | Inter-Symbol Interference | Multipath delay spread exceeding `cp_len` spills energy between OFDM symbols |
| LDPC | Low-Density Parity-Check | LDPC(174,91) in FT8/FT4 (`codec/ldpc.rs`); parameterized family (`fec/ldpc_codes.rs`) for COFDM |
| LEO | Low Earth Orbit | High-Doppler satellite case motivating OFDM's opt-in `PerSymbolPilotInterp` equalizer |
| LFSR | Linear-Feedback Shift Register | Basis of the `PnScrambler` whitener (`fec/scrambler.rs`) |
| LLR | Log-Likelihood Ratio | `log(P(bit=0)/P(bit=1))`; positive ↔ bit more likely 0 |
| LO | Local Oscillator | Receiver frequency reference; source of frequency offset |
| LP | Low-Pass | `FirLowpass`, `LpCascade` filter types |
| MAC | Medium Access Control | The COFDM frame layer (`FramePacket`, `OfdmFrameMod`/`OfdmFrameStreamDemod`) |
| MCS | Modulation and Coding Scheme | `McsTable` maps a per-frame index to (constellation, inner/outer FEC) |
| ML | Maximum Likelihood | The van de Beek guard-interval timing/CFO estimator (`dvb_t_gi_sync`); not *max-log* (LLR approximation) |
| MLSE | Maximum-Likelihood Sequence Estimation | `viterbi_decode_coherent` in `codec/psk31.rs` |
| MMSE | Minimum Mean-Square Error | Noise-aware equalizer; a candidate remedy for high-order-QAM multipath (not yet implemented) |
| MPEG-TS | MPEG Transport Stream | The 188-byte-packet payload DVB-T carries; the RS(204,188) code protects one TS packet |
| NB-DVB-T | Narrowband DVB-T | DVB-T on amateur bands (DATV), fs-scaled to 333 kHz/1 MHz/2 MHz; same 2K structure |
| NBFM | Narrowband FM | Voice FM with a small deviation-to-audio-bandwidth ratio |
| NCO | Numerically Controlled Oscillator | `Nco` in `dsp/nco.rs`; phasor recurrence |
| OFDM | Orthogonal Frequency-Division Multiplexing | `multicarrier/` + `modulate`/`demodulate`/`sync::ofdm*`; VHF–EHF target bands |
| OTFS | Orthogonal Time Frequency Space | Planned future `multicarrier/`-based waveform |
| PLL | Phase-Locked Loop | PSK31's AFC loop is a first-order decision-directed PLL |
| PM | Phase Modulation | Quadrature (dφ) demod |
| PN | Pseudo-Noise | Deterministic pseudo-random sequence; the `PnScrambler` whitener and S&C preamble base sequence |
| PRBS | Pseudo-Random Binary Sequence | DVB-T's w_k generator (X^11+X^2+1) for pilot values and the TPS DBPSK reference |
| QAM | Quadrature Amplitude Modulation | 16/64/256-QAM implemented |
| QPSK | Quadrature Phase-Shift Keying | 2 bits/symbol |
| RF | Radio Frequency | Upconverted (non-baseband) signal |
| Roll-off | (symbol-window roll-off) | Raised-cosine taper length per symbol edge (samples), set on `CarrierPlan::with_window_roll_off`; the `β` fraction forms give `round(β·cp_len)` or `round(β·n_fft)`. See [ofdm.md](ofdm.md) |
| RMS | Root Mean Square | Used by AGC and test SNR helpers |
| RS | Reed–Solomon | Byte-symbol block code (`fec/reed_solomon.rs`); DVB-T RS(204,188) t=8; COFDM outer FEC option |
| RX | Receive / Receiver | — |
| S&C | Schmidl & Cox | Repeated-segment preamble algorithm used by `ofdm_sync` for timing/CFO |
| SC-FDMA | Single-Carrier Frequency-Division Multiple Access | DFT-s-OFDM; planned future `multicarrier/`-based waveform |
| SDR | Software-Defined Radio | — |
| SNR | Signal-to-Noise Ratio | Expressed in dB throughout |
| SSB | Single-Sideband | Phasing (Weaver) modulator; product demodulator |
| TDF-II | Transposed Direct Form II | Biquad filter state-variable structure |
| TPS | Transmission Parameter Signalling | DVB-T/DVB-H signalling on 17 carriers, DBPSK over a 68-symbol frame; `HeaderFormat::DvbTps` |
| Tukey | Tukey (tapered-cosine) window | The raised-cosine symbol-edge taper `SymbolWindow` applies for TX out-of-band suppression; RX-transparent via the FFT-window back-off. See [ofdm.md](ofdm.md) |
| TX | Transmit / Transmitter | — |
| UHF | Ultra High Frequency | 300 MHz–3 GHz; secondary target band |
| VHF | Very High Frequency | 30–300 MHz; lower end of the OFDM target-band range |
| WBFM | Wideband FM | Broadcast FM with a large deviation-to-audio-bandwidth ratio |
| ZF | Zero-Forcing | `OfdmEqualizer`'s per-bin channel-inverse equalization; amplifies noise at spectral nulls |
