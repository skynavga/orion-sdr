# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Changelog

- v0.0.4: update roadmap
- v0.0.3: update description
- v0.0.2: add API implementation and basic test
- v0.0.1: placeholder API (`version()`), project structure, roadmap.

## Status (as of v0.0.3)

-  Core trait `Block`, runner `run_block` ✅
-  NCO, FIR, DC blocker ✅
-  SSB product detector ✅
-  PyO3 binding for SSB and simple Python `process` ✅
-  Basic DSP test: audio tone detection at 1 kHz ✅

## Next Milestones

- Add AGC, decimators
- Implement CW & envelope AM demods
- Build graph scheduler API
- Expose full pipeline via Python, record/replay, UI, etc.
