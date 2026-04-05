# Features (as of v0.0.27)

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
- PSK31 full stack (BPSK31 + QPSK31):
  - Varicode codec: IZ8BLY/G3PLX table, `VaricodeEncoder`, `VaricodeDecoder`
  - Convolutional codec: rate-1/2 K=5 encoder + soft Viterbi (batch and streaming) (`conv_encode`, `viterbi_decode`, `StreamingViterbi`)
  - Waveform mod/demod: Hann-windowed DBPSK/DQPSK at 31.25 baud with decision-feedback
    matched filtering and AFC (`Bpsk31Mod`, `Bpsk31Demod`, `Qpsk31Mod`, `Qpsk31Demod`)
  - Carrier sync: waterfall energy-persistence search (`psk31_sync`)
- Unit, roundtrip, and throughput tests
- Python bindings (35 classes/functions total, including full PSK31 stack)
