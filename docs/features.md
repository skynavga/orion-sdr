# Features (as of v0.0.16)

- Core traits and runner
- Basic, IQ→IQ, IQ→Audio, Audio→IQ graph schedulers
- NCO, Phase Rotator, IIR/FIR low-pass, DC blocker, FIR decimator, AGC, IIR cascade
- CW, AM, SSB, FM, PM modulators and demodulators
- BPSK, QPSK, QAM-16/64/256 modulators and demodulators
- FT8/FT4 full stack:
  - CPFSK waveform mod/demod (`Ft8Mod`, `Ft8Demod`, `Ft4Mod`, `Ft4Demod`)
  - Channel codec: CRC-14 + LDPC(174,91) + Gray code (`Ft8Codec`, `Ft4Codec`)
  - Frame sync: Costas-array waterfall search, soft-LLR extraction (`ft8_sync`, `ft4_sync`)
  - Message packing: standard QSO, free text, telemetry, nonstandard callsigns (`pack77`/`unpack77`)
- Unit, roundtrip, and throughput tests
- Python bindings (28 classes/functions total, including full FT8/FT4 stack)
