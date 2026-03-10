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
    pm.rs             — PmQuadratureDemod
    qam.rs            — QamDemod, QamDecider<BITS>, Qam16/64/256Decider
    qpsk.rs           — QpskDemod, QpskDecider
    ssb.rs            — SsbProductDemod
  modulate/
    am.rs             — AmDsbMod
    bpsk.rs           — BpskMapper, BpskMod
    cw.rs             — CwKeyedMod
    fm.rs             — FmPhaseAccumMod
    pm.rs             — PmDirectPhaseMod
    qam.rs            — QamMapper<BITS>, QamMod, Qam16/64/256Mapper
    qpsk.rs           — QpskMapper, QpskMod
    ssb.rs            — SsbPhasingMod
  python/
    mod.rs            — PyO3 module entry point, class registration
    demodulate.rs     — Python wrappers for all 8 demodulators (5 analog + 3 digital)
    modulate.rs       — Python wrappers for all 8 modulators (5 analog + 3 digital)
  tests/
    unit.rs           — unit tests
    roundtrip.rs      — mod→demod SNR roundtrip tests
    throughput.rs     — throughput benchmarks (feature-gated)
```
