// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/dvb_t_ts.rs
//
// DVB-T MPEG-2 transport-stream (TS) payload adaptation and energy dispersal,
// ETSI EN 300 744 §4.3.1. Real (NB-)DVB-T carries a genuine 188-byte MPEG-2 TS —
// this module models that packet structure so the DVB-T pipeline ingests TS
// packets rather than opaque bytes, applying the standard's exact
// energy-dispersal rules that key off packet boundaries:
//
//   • Packets are 188 bytes: one sync byte (0x47) + 187 payload bytes.
//   • The PRBS (1 + X^14 + X^15, init 100101010000000) re-initializes at the
//     start of every group of EIGHT packets.
//   • The sync byte of the FIRST packet in each group is inverted 0x47 → 0xB8
//     (the descrambler's re-init signal). The other seven packets keep 0x47.
//   • Sync bytes are NOT randomized, but the PRBS generator keeps clocking over
//     them (8 steps, output discarded) so the register phase stays aligned; the
//     187 payload bytes of every packet ARE randomized.
//
// This wraps the Phase-1 `DvbTEnergyDispersal` whitener (which is the bit-exact
// PRBS) with the packet framing. The RS(204,188) outer code then protects each
// randomized 188-byte packet (sync byte included) — one TS packet is exactly one
// RS information block, which is why the payload FEC needs no stuffing.

use crate::waveform::dvb_t::DvbTEnergyDispersal;

/// MPEG-2 transport-stream packet length in bytes (1 sync + 187 payload).
pub const TS_PACKET_LEN: usize = 188;
/// Number of payload bytes per TS packet (all but the sync byte).
pub const TS_PAYLOAD_LEN: usize = TS_PACKET_LEN - 1;
/// The MPEG-2 sync byte (`0x47`).
pub const TS_SYNC_BYTE: u8 = 0x47;
/// The inverted sync byte (`0xB8`) marking the first packet of an 8-packet group.
pub const TS_SYNC_BYTE_INVERTED: u8 = 0xB8;
/// Number of TS packets per energy-dispersal group (PRBS re-init period).
pub const TS_DISPERSAL_GROUP: usize = 8;

/// Applies (or inverts) DVB-T energy dispersal over a whole number of 188-byte TS
/// packets, in place. Self-inverse: running scrambled packets through again
/// recovers the originals, because the sync-byte inversion is deterministic per
/// group position and the PRBS is the same data-independent sequence.
///
/// `packets` must be a multiple of [`TS_PACKET_LEN`] bytes and each packet must
/// begin with a sync byte (`0x47` or its inverted form `0xB8`). Per the standard:
/// the PRBS re-inits every 8 packets, the first packet of each group has its sync
/// byte inverted, and the PRBS clocks over every sync byte without randomizing
/// it. Returns the number of packets processed.
///
/// # Panics
///
/// Panics if `packets.len()` is not a multiple of [`TS_PACKET_LEN`].
pub fn ts_energy_disperse(packets: &mut [u8]) -> usize {
    assert_eq!(
        packets.len() % TS_PACKET_LEN,
        0,
        "TS energy dispersal needs whole 188-byte packets"
    );
    let n_packets = packets.len() / TS_PACKET_LEN;
    let mut prbs = DvbTEnergyDispersal::new();
    for (i, packet) in packets.chunks_mut(TS_PACKET_LEN).enumerate() {
        let group_pos = i % TS_DISPERSAL_GROUP;
        if group_pos == 0 {
            // New 8-packet group: re-init the PRBS and toggle the sync byte to
            // its inverted form (self-inverse: 0x47 ↔ 0xB8). The first PRBS output
            // bit lands on the MSB of the byte immediately AFTER this inverted
            // sync byte — so the generator is NOT clocked over it.
            prbs.reset();
            packet[0] ^= TS_SYNC_BYTE ^ TS_SYNC_BYTE_INVERTED; // flip 0x47<->0xB8
        } else {
            // Subsequent packets in the group: the sync byte is left unrandomized
            // but the PRBS keeps clocking over it (8 steps, output discarded).
            prbs.advance_byte();
        }
        // Payload: randomized MSB-first, register carried across packets.
        prbs.feed_in_place(&mut packet[1..]);
    }
    n_packets
}

/// Wraps arbitrary payload bytes into whole TS packets (sync byte + 187 payload),
/// zero-padding the final packet's payload. This is a minimal TS adaptation for
/// the library's own end-to-end use (not a full MPEG-2 multiplexer): every packet
/// gets a plain `0x47` sync byte; `ts_energy_disperse` later inverts the
/// group-leading ones. Returns the packetized byte stream (a multiple of
/// [`TS_PACKET_LEN`]).
pub fn ts_packetize(payload: &[u8]) -> Vec<u8> {
    let n_packets = payload.len().div_ceil(TS_PAYLOAD_LEN).max(1);
    let mut out = vec![0u8; n_packets * TS_PACKET_LEN];
    for (p, chunk) in payload.chunks(TS_PAYLOAD_LEN).enumerate() {
        let base = p * TS_PACKET_LEN;
        out[base] = TS_SYNC_BYTE;
        out[base + 1..base + 1 + chunk.len()].copy_from_slice(chunk);
    }
    // If payload was empty, still emit one all-zero-payload packet with a sync.
    if payload.is_empty() {
        out[0] = TS_SYNC_BYTE;
    }
    out
}

/// Recovers the payload bytes from whole TS packets (inverse of [`ts_packetize`]):
/// strips each packet's sync byte and concatenates the 187-byte payloads. The
/// caller trims trailing zero padding via the known original length. Returns
/// `None` if `packets` is not a whole number of TS packets.
pub fn ts_depacketize(packets: &[u8]) -> Option<Vec<u8>> {
    if packets.is_empty() || !packets.len().is_multiple_of(TS_PACKET_LEN) {
        return None;
    }
    let mut out = Vec::with_capacity(packets.len() / TS_PACKET_LEN * TS_PAYLOAD_LEN);
    for packet in packets.chunks(TS_PACKET_LEN) {
        out.extend_from_slice(&packet[1..]);
    }
    Some(out)
}
