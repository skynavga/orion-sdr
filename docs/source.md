<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Source Layout

```text
python/
  orion_sdr/
    __init__.py       — re-exports from native extension (hand-maintained allowlist)
    __init__.pyi      — hand-authored PEP 561 type stub
    py.typed          — PEP 561 marker
  tests/
    __init__.py
    ...               — pytest tests for the Python interface (includes test_ofdm.py)

src/
  lib.rs              — crate root, public API re-exports
  core.rs             — Block trait, WorkReport, chain schedulers
                        (AudioToIqChain, IqToIqChain, IqToAudioChain)
  util.rs             — rms, tone, snr_db_at, atan2_approx, run_block helpers,
                        power_spectrum, nb_spectrum_snr_db, wb_spectrum_snr_db,
                        spectrum_bw_hz, best_sync, SIGNAL_THRESHOLD, PSK31_BW_HZ
  dsp/
    agc.rs            — AgcRms, AgcRmsIq
    dc.rs             — DcBlocker (1st-order HP: y = x - x1 + r·y1)
    decim.rs          — FirDecimator
    fir.rs            — FirLowpass, HalfCosineMf
    iir.rs            — Biquad, LpCascade, LpDcCascade
    nco.rs            — Nco, mix_with_nco
    rotator.rs        — Rotator
  multicarrier/        — waveform-agnostic FFT-domain primitives, shared by OFDM
                        and future multicarrier waveforms (SC-FDMA, OTFS)
    config.rs          — SubcarrierRole, CarrierPlan, CarrierPlanError
    cyclic_prefix.rs    — CyclicPrefixInsert, CyclicPrefixRemove
    fft.rs              — FftBlock, IfftBlock (allocation-free, cached rustfft plan)
    grid.rs             — CarrierGrid, GridMap, GridExtract
  demodulate/
    am.rs             — AmEnvelopeDemod (PowerSqrt, AbsApprox)
    bpsk.rs           — BpskDemod, BpskDecider
    cw.rs             — CwEnvelopeDemod
    fm.rs             — FmQuadratureDemod
    ft4.rs            — Ft4Demod (Goertzel energy per tone per symbol)
    ft8.rs            — Ft8Demod (same, 8 tones)
    ofdm.rs            — OfdmDemod, OfdmDecider, OfdmEqualizer, EqualizerMethod,
                        OfdmSoftDemod, OfdmRxFrame, build_ofdm_rx_frame,
                        bpsk_soft_llr/qpsk_soft_llr/qam_soft_llr
    ofdm_frame.rs      — OfdmFrameDemod (batch), OfdmFrameStreamDemod (streaming
                        feed/flush RX), RxFrame (COFDM frame demodulator)
    dvb_t_frame.rs     — DvbTFrameDemod, DvbTRxFrame (conformant preamble-less
                        DVB-T RX: GI-acquire → equalize → soft-demap → TPS + FEC;
                        optional integer-CFO builder flag)
    pm.rs             — PmQuadratureDemod
    psk31.rs          — Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider
    qam.rs            — QamDemod, QamDecider<BITS>, Qam16/64/256Decider
    qpsk.rs           — QpskDemod, QpskDecider
    ssb.rs            — SsbProductDemod
  modulate/
    am.rs             — AmDsbMod
    bpsk.rs           — BpskMapper, BpskMod
    cw.rs             — CwKeyedMod
    fm.rs             — FmPhaseAccumMod
    ft4.rs            — Ft4Mod, Ft4Frame (CPFSK, 4-FSK, 12 kHz)
    ft8.rs            — Ft8Mod, Ft8Frame (CPFSK, 8-FSK, 12 kHz)
    ofdm.rs            — ConstellationOrder, OfdmConfig (incl. COFDM frame-layer
                        FEC/interleaver/CRC/scrambler/header fields), OfdmMod
                        (mapper → grid → IFFT → cyclic prefix → optional RF)
    ofdm_frame.rs      — OfdmFrameMod, Mcs, McsTable (COFDM frame modulator +
                        shared coding chain and block-size bookkeeping)
    dvb_t_frame.rs     — DvbTFrameMod, DvbTFrame (conformant preamble-less
                        DVB-T TX: TS + energy dispersal → FEC → map → scattered grid + TPS)
    pm.rs             — PmDirectPhaseMod
    psk31.rs          — Bpsk31Mod, Qpsk31Mod (DBPSK/DQPSK, 31.25 baud, Hann pulse shaping)
    qam.rs            — QamMapper<BITS>, QamMod, Qam16/64/256Mapper
    qpsk.rs           — QpskMapper, QpskMod
    ssb.rs            — SsbPhasingMod
  codec/
    crc.rs            — ft8_crc14, ft8_add_crc, ft8_extract_crc (poly 0x2757)
    ft4.rs            — Ft4Codec: same interface; includes FT4 XOR scramble
    ft8.rs            — Ft8Codec: encode, decode_hard, decode_soft, frame_to_llr_hard
    gray.rs           — gray8_encode/decode (FT8), gray4_encode/decode (FT4)
    ldpc.rs           — ldpc_encode, ldpc_decode_soft (LDPC(174,91), BP 20 iter)
    morse.rs          — MorseEncoder (ITU-R M.1677 keying-envelope generator for CW)
    psk31.rs          — conv_encode, viterbi_decode, viterbi_decode_coherent, StreamingViterbi,
                        Psk31Stream (streaming BPSK31/QPSK31 decode pipeline)
    varicode.rs       — varicode_encode/decode, VaricodeEncoder, VaricodeDecoder (IZ8BLY)
  fec/                 — waveform-agnostic channel-coding + framing primitives for
                        the COFDM (coded OFDM) frame layer
    gf.rs              — Gf256 (GF(2^8) log/antilog arithmetic; shared by BCH, RS)
    ldpc_codes.rs      — Ldpc, LdpcCode (parameterized fixed-family LDPC: staircase H,
                        sum-product decode; distinct from codec/ldpc.rs's FT8 code)
    bch.rs             — Bch (binary BCH(n,k,t) over GF(2^8); BM + Chien decode)
    reed_solomon.rs    — ReedSolomon (RS(204,188) t=8 and shortened variants;
                        syndrome → BM → Chien → Forney)
    conv.rs            — PunctureRate, conv_encode_punctured, viterbi_decode_soft
                        (zero-tail punctured convolutional; LLR-domain soft Viterbi)
    interleaver.rs     — BlockInterleaver (rectangular, generic over bits and LLRs)
    scrambler.rs       — PnScrambler (parameterized additive LFSR whitener)
    frame.rs           — FramePacket, FrameMetadata, RxError, and the scheme
                        descriptors (OuterFec, InnerFec, InterleaverKind, CrcKind,
                        ScramblerKind/SeedMode/ScramblerPos, HeaderFormat)
  waveform/            — standard-specific waveform assemblies built from the generic
                        fec/multicarrier/modulate/demodulate primitives (per-standard
                        parameter tables + orchestrators; no new generic abstractions)
    dvb_t.rs           — DVB-T 2K numerology, fs-scaling, continual/scattered pilots,
                        Figure-9a mapping + soft-LLR, energy-dispersal PRBS, MCS table,
                        scattered-pilot orchestrators, DvbTFrameParams (TX/RX-shared)
    dvb_t_ts.rs        — MPEG-2 TS packet adaptation + energy dispersal (188-byte
                        packets, 8-packet PRBS re-init, sync-byte inversion)
    dvb_t_tps.rs       — TPS signalling: standalone GF(2^7) BCH(67,53), TpsWord
                        pack/unpack, DBPSK-along-symbol-axis TpsEncoder/TpsDecoder
  sync/
    costas.rs         — Costas difference-metric scorer and top-N candidate search
    ft4_sync.rs       — ft4_sync() → Vec<Ft4SyncResult>
    ft8_sync.rs       — ft8_sync() → Vec<Ft8SyncResult> (each has .llr: [f32; 174])
    ofdm_sync.rs       — OfdmPreamble, TrainingSymbol, OfdmSyncResult, ofdm_sync(),
                        generate_ofdm_preamble() (Schmidl & Cox timing/fractional-CFO,
                        integer-CFO via training-symbol correlation)
    dvb_t_gi_sync.rs   — dvb_t_gi_sync(), GiSyncConfig (van de Beek guard-interval ML
                        timing/CFO for preamble-less OFDM: |γ| − ρ·Φ over the cyclic prefix)
    psk31_sync.rs     — psk31_sync() → Vec<Psk31SyncResult> (energy persistence carrier search)
    waterfall.rs      — symbol-rate magnitude spectrogram (Goertzel per tone per symbol)
  message/
    tables.rs         — Table enum, nchar/charn (6 char tables matching ft8_lib)
    callsign.rs       — pack_basecall, pack28/unpack28, pack58/unpack58,
                        CallsignHashTable (22-bit multiply-shift hash)
    grid.rs           — packgrid/unpackgrid, GridField enum
    free_text.rs      — encode_free_text/decode_free_text (base-42, 71-bit)
    message.rs        — Ft8Message enum, pack77/unpack77
  python/               — PyO3 bindings, cfg-gated on the `extension-module` feature
    mod.rs            — PyO3 module entry point, class/function registration
    demodulate.rs     — Python wrappers for analog + BPSK/QPSK/QAM demodulators
    ft8.rs            — Python wrappers for FT8/FT4 waveform, codec, sync, and message
    modulate.rs       — Python wrappers for analog + BPSK/QPSK/QAM modulators
    ofdm.rs            — Python wrappers for OFDM config, mod, demod, RX frame, sync
    ofdm_frame.rs      — Python wrappers for the COFDM frame layer (FramePacket,
                        McsTable, OfdmFrameMod, OfdmFrameStreamDemod, OfdmFrameDemod)
    psk31.rs          — Python wrappers for PSK31 mod/demod, Varicode, and psk31_sync

tests/
  common/mod.rs       — shared test helpers (snr_db_at, add_awgn)
  unit.rs             — entry point for unit tests
  unit/               — per-module unit tests (one file per modulation type, plus
                        multicarrier.rs, ofdm.rs, ofdm_sync.rs for OFDM)
  roundtrip.rs        — entry point for roundtrip tests
  roundtrip/          — mod→demod→decode full-stack tests (one file per mode, plus
                        ofdm.rs and ofdm_snr.rs for OFDM's CI BER-threshold tests)
  performance.rs      — entry point for performance tests
  performance/
    throughput/        — throughput benchmarks (feature-gated, one file per mode,
                        plus multicarrier.rs and ofdm.rs)
    snr/              — SNR/acquisition sensitivity sweeps (feature-gated, includes
                        ofdm.rs and ofdm_sync.rs)
```
