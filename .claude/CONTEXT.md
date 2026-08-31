# orion-sdr

A composable SDR/DSP library (Rust ed. 2024) spanning HF-through-EHF signal
processing, with Python bindings via PyO3.

## Language

**MCS (DVB-T)**:
For DVB-T, the TPS-signalled tuple of (constellation, HP inner code rate, LP
inner code rate under hierarchical modulation). Distinct from the crate's
generic `McsTable` (see [terminology.md](docs/terminology.md)), which indexes
a table for the COFDM MAC frame layer — DVB-T signals these fields directly
via TPS bits, not through a table index, and its outer RS(204,188) code is
fixed rather than part of the signalled scheme. MCS changes apply one
superframe after they're advertised in TPS, never mid-superframe.
_Avoid_: using "MCS" to also cover guard interval or transmission mode.

**Waveform configuration (DVB-T)**:
Guard interval and transmission mode (2K/8K/4K) — also TPS-signalled with the
same one-superframe advance notice as MCS, but orthogonal to it: guard
interval only affects symbol timing (`cp_len`), and transmission mode changes
the entire carrier/pilot table set. Neither affects which carriers are pilots
for a fixed transmission mode, nor how data carriers get demapped — that's
MCS's job.
_Avoid_: conflating with MCS.
