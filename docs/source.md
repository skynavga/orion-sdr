# Source Layout

```text
python/
  orion_sdr/
    __init__.py       — re-exports from native extension
    __init__.pyi      — hand-authored PEP 561 type stub
    py.typed          — PEP 561 marker
  tests/
    __init__.py
    ...               — pytest tests for the Python interface

src/
  lib.rs              — crate root, public API re-exports
  core.rs             — Block trait, WorkReport, chain schedulers
                        (IqToAudioChain, IqToIqChain, AudioToIqChain, BasicChain)
  util.rs             — rms, tone, snr_db_at, atan2_approx, run_block helpers
  dsp/
    agc.rs            — AgcRms, AgcRmsIq
    dc.rs             — DcBlocker (1st-order HP: y = x - x1 + r·y1)
    decim.rs          — FirDecimator
    fir.rs            — FirLowpass
    iir.rs            — Biquad, LpCascade, LpDcCascade
    nco.rs            — Nco, mix_with_nco
    rotator.rs        — Rotator
  demodulate/
    am.rs             — AmEnvelopeDemod (PowerSqrt, AbsApprox)
    bpsk.rs           — BpskDemod, BpskDecider
    cw.rs             — CwEnvelopeDemod
    fm.rs             — FmQuadratureDemod
    ft4.rs            — Ft4Demod (Goertzel energy per tone per symbol)
    ft8.rs            — Ft8Demod (same, 8 tones)
    pm.rs             — PmQuadratureDemod
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
    pm.rs             — PmDirectPhaseMod
    qam.rs            — QamMapper<BITS>, QamMod, Qam16/64/256Mapper
    qpsk.rs           — QpskMapper, QpskMod
    ssb.rs            — SsbPhasingMod
  codec/
    crc.rs            — ft8_crc14, ft8_add_crc, ft8_extract_crc (poly 0x2757)
    ldpc.rs           — ldpc_encode, ldpc_decode_soft (LDPC(174,91), BP 20 iter)
    gray.rs           — gray8_encode/decode (FT8), gray4_encode/decode (FT4)
    ft8.rs            — Ft8Codec: encode, decode_hard, decode_soft, frame_to_llr_hard
    ft4.rs            — Ft4Codec: same interface; includes FT4 XOR scramble
  sync/
    waterfall.rs      — symbol-rate magnitude spectrogram (Goertzel per tone per symbol)
    costas.rs         — Costas difference-metric scorer and top-N candidate search
    ft8_sync.rs       — ft8_sync() → Vec<Ft8SyncResult> (each has .llr: [f32; 174])
    ft4_sync.rs       — ft4_sync() → Vec<Ft4SyncResult>
  message/
    tables.rs         — Table enum, nchar/charn (6 char tables matching ft8_lib)
    callsign.rs       — pack_basecall, pack28/unpack28, pack58/unpack58,
                        CallsignHashTable (22-bit multiply-shift hash)
    grid.rs           — packgrid/unpackgrid, GridField enum
    free_text.rs      — encode_free_text/decode_free_text (base-42, 71-bit)
    message.rs        — Ft8Message enum, pack77/unpack77
  python/
    mod.rs            — PyO3 module entry point, class registration
    demodulate.rs     — Python wrappers for demodulators
    modulate.rs       — Python wrappers for modulators
    ft8.rs            — Python wrappers for FT8/FT4 waveform, codec, sync, and message
  tests/
    unit/             — per-module unit tests (one file per module)
    roundtrip/        — mod→demod→decode full-stack tests (one file per mode)
    performance/
      throughput/     — throughput benchmarks (feature-gated, one file per mode)
      snr/            — SNR sensitivity sweeps (feature-gated, always pass, print curve)
```
