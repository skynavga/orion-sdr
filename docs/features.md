<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Features (as of v0.0.41)

- Core `Block` trait and runner
- IQ→IQ, Audio→IQ, IQ→Audio graph schedulers (`IqToIqChain`, `AudioToIqChain`, `IqToAudioChain`)
- NCO, Phase Rotator, IIR/FIR low-pass, DC blocker, FIR decimator, AGC, IIR cascade
- CW, AM, SSB, FM, PM modulators and demodulators
- BPSK, QPSK, QAM-16/64/256 modulators and demodulators
- FT8/FT4 full stack:
  - CPFSK waveform mod/demod (`Ft8Mod`, `Ft8Demod`, `Ft4Mod`, `Ft4Demod`)
  - Channel codec: CRC-14 + LDPC(174,91) + Gray code (`Ft8Codec`, `Ft4Codec`)
  - Frame sync: Costas-array waterfall search, soft-LLR extraction (`ft8_sync`, `ft4_sync`)
  - Message packing: standard QSO, free text, telemetry, nonstandard callsigns (`pack77`/`unpack77`)
- PSK31 full stack (BPSK31 + QPSK31):
  - Varicode codec: IZ8BLY/G3PLX table, `VaricodeEncoder`, `VaricodeDecoder`
  - Convolutional codec: rate-1/2 K=5 encoder + soft Viterbi (batch and streaming) (`conv_encode`, `viterbi_decode`, `StreamingViterbi`)
  - Waveform mod/demod: Hann-windowed DBPSK/DQPSK at 31.25 baud with decision-feedback
    matched filtering and AFC (`Bpsk31Mod`, `Bpsk31Demod`, `Qpsk31Mod`, `Qpsk31Demod`)
  - Carrier sync: waterfall energy-persistence search (`psk31_sync`)
- OFDM full stack, targeting VHF through EHF (predominantly line-of-sight
  terrestrial-microwave and satellite links); first of a planned multicarrier
  family sharing the `multicarrier/` module (DFT-s-OFDM/SC-FDMA and OTFS to follow):
  - Waveform-agnostic FFT-domain primitives: allocation-free `FftBlock`/`IfftBlock`,
    `CyclicPrefixInsert`/`CyclicPrefixRemove`
  - Resource-grid mapping: `CarrierPlan` (caller-owned numerology), `CarrierGrid`
    (signed carrier-index → FFT-bin resolution), `GridMap`/`GridExtract`
  - Waveform mod/demod: `OfdmMod` (mapper → grid → IFFT → CP → optional RF
    upconversion), `OfdmDemod`/`OfdmDecider` (inverse chain + hard decision)
  - Packet sync + CFO acquisition: Schmidl & Cox repeated-segment preamble
    (`ofdm_sync`, `generate_ofdm_preamble`), fractional CFO (±½ subcarrier
    spacing) plus wide-range integer-CFO recovery via a shared training symbol
  - Channel estimation + equalization: `OfdmEqualizer` with `TrainingSymbolHold`
    (default, one estimate/packet) and `PerSymbolPilotInterp` (opt-in, for
    time-varying/Doppler channels) methods
  - Soft (LLR) demapping per constellation order (`OfdmSoftDemod`,
    `bpsk_soft_llr`/`qpsk_soft_llr`/`qam_soft_llr`); no mandatory FEC — soft
    LLRs are the deliverable for an external/user-supplied FEC layer
  - Per-packet RX diagnostics (`OfdmRxFrame`: EVM, CFO, timing offset, channel MSE)
- Unit, roundtrip, throughput, and SNR-sensitivity tests (212 default `cargo
  test --release`, 254 total including `--features throughput`)
- Python bindings (45 classes/functions total, including full PSK31 and OFDM stacks)
