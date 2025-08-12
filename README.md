# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Changelog

- v0.0.5: add graph scheduler; AGC, FIR decimator; CW and AM demods; more tests
- v0.0.4: update roadmap
- v0.0.3: update description
- v0.0.2: add API implementation and basic test
- v0.0.1: placeholder API (`version()`), project structure, roadmap.

## Status (as of v0.0.5)

- Core trait `Block`, runner `run_block` ✅
- Basic graph scheduler ✅
- NCO, FIR low pass, DC blocker, FIR decimator, AGC ✅
- CW, AM, SSB demods ✅
- PyO3 binding for SSB and simple Python `process` ✅
- Basic DSP tests ✅

## Next Milestones

- Expose full pipeline via Python, record/replay, UI, etc.
