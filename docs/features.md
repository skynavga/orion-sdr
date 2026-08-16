<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Features

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
    time-varying/Doppler channels) methods; a bin whose estimate falls under the
    null floor is erased rather than divided through, handing the FEC an erasure
    instead of a confidently-wrong large value
  - Soft (LLR) demapping per constellation order (`OfdmSoftDemod`,
    `bpsk_soft_llr`/`qpsk_soft_llr`/`qam_soft_llr`)
  - Per-packet RX diagnostics (`OfdmRxFrame`: EVM, CFO, timing offset, channel MSE)
- COFDM (coded, framed OFDM) frame/MAC layer built on the OFDM PHY
  (see [ofdm.md](ofdm.md)):
  - Waveform-agnostic FEC/framing primitives (`fec/`): parameterized LDPC and
    punctured convolutional inner codes, BCH and Reed–Solomon (DVB-T RS(204,188))
    outer codes over GF(2^8), block interleaver, PN scrambler, generic CRC-16/32,
    and the `FramePacket`/`RxError` types
  - Concatenated FEC configured on `OfdmConfig` (outer/inner code, two
    interleavers, scrambler + position, header/payload CRC, header format), all
    defaulted-off and set via `with_*` builders; per-frame adaptive coding via
    an `McsTable`
  - Selectable LDPC check-node decode rule (`with_ldpc_decode_rule`): exact
    sum-product (default) or opt-in min-sum / scaled-min-sum (~2× decode for a
    sub-0.3 dB coding-gain cost on the payload; the header always uses sum-product)
  - Per-link `CodecCache` sharing constructed FEC codes across frames (and,
    optionally, across a modulator/demodulator pair), with a process-wide
    `Gf256`; every frame object owns a persistent cache or accepts a shared one
  - Frame TX (`OfdmFrameMod`) and streaming RX (`OfdmFrameStreamDemod`,
    feed/flush) with in-band header, acquisition, CFO correction, equalization,
    concatenated decode, and CRC check; batch `OfdmFrameDemod` for a known start
  - Band-limited preamble: the S&C repeats are built in the frequency domain so
    the repetition is exact by construction and the preamble stays inside the
    payload's band, cutting ~45 dB of out-of-band excess
  - Per-frame receiver diagnostics on `OfdmRxFrame` — acquisition score, CFO and
    timing offset, EVM, and inner/outer FEC convergence reported separately —
    plus two opt-ins: the per-bin channel estimate (`with_channel_estimate`,
    the basis for a power delay profile) and true pre/post-inner-FEC bit error
    rates (`with_error_rates`), measured by re-encoding a decoded frame so no
    prior knowledge of the payload is required
  - RX probe (`feed_probed`/`flush_probed` → `OfdmRxProbe`): the equalizer's
    output symbols as the demapper saw them, plus a per-coded-bit correction map
    (`Clean`/`Corrected`/`Uncorrected`/`Introduced`) that is the exact per-bit
    expansion of `channel_ber`. Caller-owned reusable buffers, ~+3..4% of the
    receive path when on and no branch at all when off
- Out-of-band spectral shaping — three independent, off-by-default levers that
  compose, with a shared cyclic-prefix budget (see [ofdm.md](ofdm.md) for the
  geometry, [modulate.md](modulate.md#choosing-the-numbers) for sizing):
  - Edge-carrier guard band (COFDM only): `CarrierPlan::with_contiguous_data`
    nulls `edge_guard` carriers per band edge, moving the loudest `sinc`
    generators inward and creating the null band a mask filters into
  - TX symbol windowing (`SymbolWindow`): raised-cosine (Tukey) per-symbol edge
    taper, same-length and stateless; an `orion-sdr` original, **not** a DVB
    standard mechanism, so RX-transparent by construction
  - TX baseband spectral mask (`TxLowpass` over `dsp::FirLowpassIq`): a
    Kaiser-designed linear-phase FIR across the assembled stream, applied
    group-delay-compensated — the one lever not bounded by the windowing ceiling
  - RX FFT-window back-off (`SymbolFft`): the shared enabler for both TX levers,
    capped on DVB-T at 85 samples by the scattered-pilot grid rather than by the
    guard interval
- DVB-T / NB-DVB-T full stack, conformant to ETSI EN 300 744 2K
  (see [dvb.md](dvb.md)):
  - Fixed 2K numerology (`n_fft` 2048, 1705 active carriers) with all four guard
    intervals (1/32, 1/16, 1/8, 1/4); NB-DVB-T amateur bandwidth modes
    (`NbBandwidth`: 333 kHz / 1 MHz / 2 MHz) as a pure fs-scaling of that
    structure — the baseband frame is identical
  - Pilot grid: four-phase scattered pilots, 45 continual pilots, 17 TPS
    carriers, and the constant-1512 data-carrier invariant
  - Payload FEC chain: TS packetization + energy dispersal → RS(204,188) →
    Forney convolutional interleaver (I=12) → K=7 punctured convolutional inner
    code, with Figure-9a soft-decision demapping on RX
  - TPS signalling (`TpsWord`): DBPSK woven across a 68-symbol block, BCH(67,53)
    protected, carrying constellation / code rate / guard / frame number / cell id
  - Preamble-less acquisition: van de Beek ML guard-interval timing + fractional
    CFO (`dvb_t_gi_sync`), plus opt-in continual-pilot integer-CFO correction
    (`dvb_t_integer_cfo`, a set-once builder flag)
  - Frame TX/RX (`DvbTFrameMod`/`DvbTFrameDemod`), four-frame super-frame with
    cell-id split and frame-sequence check (`DvbTSuperFrameMod`/`Demod`), and a
    streaming receiver (`DvbTFrameStreamDemod`, feed/flush)
- Unit, roundtrip, throughput, and SNR-sensitivity tests (453 default `cargo
  test --release`, 531 total including `--features throughput`)
- Python bindings (70 classes/functions total, including full PSK31, OFDM, COFDM
  frame, DVB-T frame/super-frame/streaming, and spectral-shaping stacks), with
  201 pytest tests
