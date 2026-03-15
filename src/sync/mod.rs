// src/sync/mod.rs
//
// Frame synchronisation for FT8 and FT4.
//
// Provides:
//   - `ft8_sync` — search an IQ buffer for FT8 frames, return soft LLRs
//   - `ft4_sync` — search an IQ buffer for FT4 frames, return soft LLRs
//   - `Ft8SyncResult` / `Ft4SyncResult` — per-candidate decode inputs
//   - `Waterfall` / `compute_waterfall` — low-level spectrogram (exposed for testing)
//   - `Candidate` — sync candidate (time_sym, freq_bin, score)

pub mod waterfall;
pub mod costas;
pub mod ft8_sync;
pub mod ft4_sync;

pub use waterfall::{Waterfall, compute_waterfall};
pub use costas::Candidate;
pub use ft8_sync::{ft8_sync, Ft8SyncResult};
pub use ft4_sync::{ft4_sync, Ft4SyncResult};
