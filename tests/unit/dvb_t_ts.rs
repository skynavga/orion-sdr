// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// DVB-T MPEG-2 transport-stream energy-dispersal tests (EN 300 744 §4.3.1):
// 188-byte packet framing, 8-packet PRBS re-init, first-sync-byte inversion,
// sync-bytes-skipped-but-clocked.

use orion_sdr::waveform::dvb_t_ts::{
    TS_DISPERSAL_GROUP, TS_PACKET_LEN, TS_PAYLOAD_LEN, TS_SYNC_BYTE, TS_SYNC_BYTE_INVERTED,
    ts_depacketize, ts_energy_disperse, ts_packetize,
};

fn make_packets(n: usize) -> Vec<u8> {
    // n TS packets: each a 0x47 sync + a recognizable payload pattern.
    let mut out = vec![0u8; n * TS_PACKET_LEN];
    for (p, packet) in out.chunks_mut(TS_PACKET_LEN).enumerate() {
        packet[0] = TS_SYNC_BYTE;
        for (i, b) in packet[1..].iter_mut().enumerate() {
            *b = ((p * 7 + i * 3 + 1) & 0xff) as u8;
        }
    }
    out
}

#[test]
fn ts_constants_are_standard() {
    assert_eq!(TS_PACKET_LEN, 188);
    assert_eq!(TS_PAYLOAD_LEN, 187);
    assert_eq!(TS_SYNC_BYTE, 0x47);
    assert_eq!(TS_SYNC_BYTE_INVERTED, 0xB8);
    assert_eq!(TS_DISPERSAL_GROUP, 8);
}

#[test]
fn first_sync_byte_inverted_per_group() {
    // 9 packets → two groups (8 + 1). The group-leading packets (0 and 8) get the
    // inverted sync byte; the rest keep 0x47.
    let mut packets = make_packets(9);
    ts_energy_disperse(&mut packets);
    for p in 0..9 {
        let sync = packets[p * TS_PACKET_LEN];
        if p % TS_DISPERSAL_GROUP == 0 {
            assert_eq!(sync, TS_SYNC_BYTE_INVERTED, "packet {p} leads a group");
        } else {
            assert_eq!(sync, TS_SYNC_BYTE, "packet {p} keeps 0x47");
        }
    }
}

#[test]
fn energy_dispersal_is_self_inverse() {
    let original = make_packets(10);
    let mut work = original.clone();
    ts_energy_disperse(&mut work);
    assert_ne!(work, original, "dispersal must change the data");
    ts_energy_disperse(&mut work);
    assert_eq!(work, original, "second pass recovers the original packets");
}

#[test]
fn first_payload_byte_is_prbs_known_answer() {
    // The first PRBS output bit applies to the MSB of the first byte AFTER the
    // inverted sync byte (§4.3.1). Whitening an all-zero payload therefore makes
    // that byte the standard's first PRBS byte, 0x03.
    let mut packets = vec![0u8; TS_PACKET_LEN];
    packets[0] = TS_SYNC_BYTE;
    ts_energy_disperse(&mut packets);
    assert_eq!(packets[0], TS_SYNC_BYTE_INVERTED, "leading sync inverted");
    assert_eq!(packets[1], 0x03, "first randomized byte is the PRBS 0x03");
    assert_eq!(&packets[1..5], &[0x03, 0xF6, 0x08, 0x34], "PRBS start");
}

#[test]
fn sync_bytes_clock_prbs_between_packets() {
    // The PRBS clocks over each sync byte (output disabled), so packet 2's payload
    // is randomized with the PRBS continued past packet 1 — NOT restarted. Verify
    // packet 2's first payload byte differs from packet 1's (which would be equal
    // if the PRBS restarted each packet), for identical all-zero payloads.
    let mut packets = vec![0u8; 2 * TS_PACKET_LEN];
    packets[0] = TS_SYNC_BYTE;
    packets[TS_PACKET_LEN] = TS_SYNC_BYTE;
    ts_energy_disperse(&mut packets);
    let p1_first = packets[1];
    let p2_first = packets[TS_PACKET_LEN + 1];
    assert_eq!(p1_first, 0x03);
    assert_ne!(
        p1_first, p2_first,
        "PRBS continues (clocked over the sync byte), not restarted per packet"
    );
}

#[test]
fn prbs_reinits_each_group() {
    // Packet 8 (start of the second group) must reproduce packet 0's whitening of
    // an identical all-zero payload — the PRBS re-inits every 8 packets.
    let mut packets = vec![0u8; 9 * TS_PACKET_LEN];
    for p in 0..9 {
        packets[p * TS_PACKET_LEN] = TS_SYNC_BYTE;
    }
    ts_energy_disperse(&mut packets);
    let p0 = &packets[1..TS_PACKET_LEN];
    let p8 = &packets[8 * TS_PACKET_LEN + 1..9 * TS_PACKET_LEN];
    assert_eq!(p0, p8, "group re-init reproduces the same PRBS whitening");
}

#[test]
fn packetize_depacketize_roundtrip() {
    for len in [0usize, 1, 187, 188, 500, 187 * 3] {
        let payload: Vec<u8> = (0..len).map(|i| ((i * 31 + 5) & 0xff) as u8).collect();
        let packets = ts_packetize(&payload);
        assert_eq!(packets.len() % TS_PACKET_LEN, 0, "whole packets, len={len}");
        let recovered = ts_depacketize(&packets).expect("depacketize");
        assert_eq!(&recovered[..len], &payload[..], "roundtrip len={len}");
    }
}

#[test]
fn full_chain_ts_roundtrip() {
    // packetize → disperse → (channel) → disperse → depacketize recovers payload.
    let payload: Vec<u8> = (0..900).map(|i| ((i * 13 + 7) & 0xff) as u8).collect();
    let mut packets = ts_packetize(&payload);
    ts_energy_disperse(&mut packets);
    ts_energy_disperse(&mut packets); // descramble (self-inverse)
    let recovered = ts_depacketize(&packets).unwrap();
    assert_eq!(&recovered[..payload.len()], &payload[..]);
}

#[test]
fn depacketize_rejects_partial() {
    assert!(ts_depacketize(&[]).is_none());
    assert!(ts_depacketize(&[0u8; TS_PACKET_LEN - 1]).is_none());
    assert!(ts_depacketize(&[0u8; TS_PACKET_LEN + 5]).is_none());
}
