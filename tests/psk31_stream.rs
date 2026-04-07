use num_complex::Complex32 as C32;
use orion_sdr::codec::psk31::Psk31Stream;
use orion_sdr::codec::varicode::varicode_encode;
use orion_sdr::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, psk31_sps};
use orion_sdr::Block;

const FS: f32 = 48_000.0;
const CARRIER: f32 = 1000.0;

/// Encode a message to varicode bits (MSB-first, "00" separators).
fn message_to_bits(msg: &str) -> Vec<u8> {
    let mut bits = Vec::new();
    // Preamble: 32 zeros
    bits.extend(std::iter::repeat(0u8).take(32));
    for &byte in msg.as_bytes() {
        let (cw, len) = varicode_encode(byte);
        for i in (0..len).rev() {
            bits.push(((cw >> i) & 1) as u8);
        }
        bits.push(0);
        bits.push(0);
    }
    // Postamble: 32 zeros
    bits.extend(std::iter::repeat(0u8).take(32));
    bits
}

/// Modulate bits to IQ samples using BPSK31.
fn modulate_bpsk(bits: &[u8]) -> Vec<C32> {
    let sps = psk31_sps(FS);
    let mut modulator = Bpsk31Mod::new(FS, CARRIER, 1.0);
    let mut iq = vec![C32::new(0.0, 0.0); bits.len() * sps + sps];
    let wr = modulator.process(bits, &mut iq);
    iq.truncate(wr.out_written);
    iq
}

/// Modulate bits to IQ samples using QPSK31.
fn modulate_qpsk(bits: &[u8]) -> Vec<C32> {
    let sps = psk31_sps(FS);
    let mut modulator = Qpsk31Mod::new(FS, CARRIER, 1.0);
    let mut iq = vec![C32::new(0.0, 0.0); bits.len() * sps + sps];
    let wr = modulator.process(bits, &mut iq);
    iq.truncate(wr.out_written);
    iq
}

#[test]
fn bpsk31_stream_decodes_hello() {
    let msg = "HELLO";
    let bits = message_to_bits(msg);
    let iq = modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(FS, CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}

#[test]
fn qpsk31_stream_decodes_hello() {
    let msg = "HELLO";
    let bits = message_to_bits(msg);
    let iq = modulate_qpsk(&bits);

    let mut stream = Psk31Stream::new_qpsk(FS, CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}

#[test]
fn bpsk31_stream_decodes_all_printable_ascii() {
    // All printable ASCII characters (space through ~)
    let msg: String = (0x20u8..=0x7eu8).map(|b| b as char).collect();
    let bits = message_to_bits(&msg);
    let iq = modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(FS, CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    for ch in msg.chars() {
        assert!(decoded.contains(ch),
            "missing char '{}' (0x{:02x}) in decoded output", ch, ch as u8);
    }
}

#[test]
fn qpsk31_stream_decodes_all_printable_ascii() {
    let msg: String = (0x20u8..=0x7eu8).map(|b| b as char).collect();
    let bits = message_to_bits(&msg);
    let iq = modulate_qpsk(&bits);

    let mut stream = Psk31Stream::new_qpsk(FS, CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    for ch in msg.chars() {
        assert!(decoded.contains(ch),
            "missing char '{}' (0x{:02x}) in decoded output", ch, ch as u8);
    }
}

#[test]
fn bpsk31_stream_incremental_feed() {
    // Feed in small chunks to verify incremental operation.
    let msg = "CQ CQ DE TEST";
    let bits = message_to_bits(msg);
    let iq = modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(FS, CARRIER, 1.0);
    let chunk_size = psk31_sps(FS) * 4; // ~4 symbols per chunk
    let mut decoded = String::new();
    for chunk in iq.chunks(chunk_size) {
        decoded += &stream.feed(chunk);
    }
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}

#[test]
fn fed_up_to_tracks_position() {
    let mut stream = Psk31Stream::new_bpsk(FS, CARRIER, 1.0);
    assert_eq!(stream.fed_up_to(), 0);
    stream.set_fed_up_to(1000);
    assert_eq!(stream.fed_up_to(), 1000);

    let mut stream = Psk31Stream::new_qpsk(FS, CARRIER, 1.0);
    assert_eq!(stream.fed_up_to(), 0);
    stream.set_fed_up_to(2000);
    assert_eq!(stream.fed_up_to(), 2000);
}

#[test]
fn feed_empty_returns_empty() {
    let mut stream = Psk31Stream::new_bpsk(FS, CARRIER, 1.0);
    assert!(stream.feed(&[]).is_empty());

    let mut stream = Psk31Stream::new_qpsk(FS, CARRIER, 1.0);
    assert!(stream.feed(&[]).is_empty());
}
