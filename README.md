# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Change Log

- v0.0.6: add FM and PM demods; more tests
- v0.0.5: add graph scheduler; AGC, FIR decimator; CW and AM demods; more tests
- v0.0.4: update roadmap
- v0.0.3: update description
- v0.0.2: add API implementation and basic test
- v0.0.1: placeholder API, project structure, roadmap

## Status (as of v0.0.5)

- Core traits and runner ✅
- Basic graph scheduler ✅
- NCO, FIR low pass, DC blocker, FIR decimator, AGC ✅
- CW, AM, SSB demods ✅
- PyO3 binding for SSB and simple Python process ✅
- Basic DSP tests ✅

## Next Milestones

- Expose full pipeline via Python, record/replay, UI, etc.
