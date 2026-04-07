// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "throughput")]
mod common;

#[cfg(feature = "throughput")]
#[path = "performance/throughput/mod.rs"]
mod throughput;

#[cfg(feature = "throughput")]
#[path = "performance/snr/mod.rs"]
mod snr;
