use crate::message::{CallsignHashTable, GridField, Ft8Message, pack77, unpack77};
use crate::codec::ft8::{Ft8Codec, Ft8Bits};
use crate::codec::ft4::{Ft4Codec, Ft4Bits};
use crate::modulate::{Ft8Mod, Ft4Mod};
use crate::demodulate::{Ft8Demod, Ft4Demod};

fn payload77_to_ft8bits(p: &[u8; 10]) -> Ft8Bits {
    *p
}

fn payload77_to_ft4bits(p: &[u8; 10]) -> Ft4Bits {
    *p
}

#[test]
fn full_stack_ft8_type1() {
    let mut ht = CallsignHashTable::new();
    let msg = Ft8Message::Standard {
        call_to: "KD9ABC".to_string(),
        call_de: "W9XYZ".to_string(),
        extra: GridField::Grid("FN31".to_string()),
    };

    // 1. pack77 -> Ft8Bits
    let payload = pack77(&msg, &mut ht).expect("pack77");
    let bits: Ft8Bits = payload77_to_ft8bits(&payload);

    // 2. Ft8Codec::encode -> Ft8Frame
    let frame = Ft8Codec::encode(&bits);

    // 3. Ft8Mod -> IQ samples
    let iq = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0).modulate(&frame);

    // 4. Ft8Demod -> Ft8Frame
    let frame_out = Ft8Demod::new(12_000.0, 1_000.0)
        .demodulate(&iq)
        .expect("Ft8Demod failed");

    // 5. Ft8Codec::decode_hard -> Ft8Bits
    let bits_out = Ft8Codec::decode_hard(&frame_out).expect("decode_hard failed");

    // 6. unpack77 -> Ft8Message
    let decoded = unpack77(&bits_out, &ht);
    assert_eq!(decoded, msg, "Full-stack FT8 Type1 roundtrip mismatch");
}

#[test]
fn full_stack_ft8_free_text() {
    let mut ht = CallsignHashTable::new();
    let msg = Ft8Message::FreeText("CQ DX".to_string());

    let payload = pack77(&msg, &mut ht).expect("pack77 free text");
    let bits: Ft8Bits = payload77_to_ft8bits(&payload);
    let frame = Ft8Codec::encode(&bits);
    let iq = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft8Demod::new(12_000.0, 1_000.0)
        .demodulate(&iq)
        .expect("Ft8Demod failed");
    let bits_out = Ft8Codec::decode_hard(&frame_out).expect("decode_hard failed");
    let decoded = unpack77(&bits_out, &ht);
    assert_eq!(decoded, msg, "Full-stack FT8 free text roundtrip mismatch");
}

#[test]
fn full_stack_ft4_type1() {
    let mut ht = CallsignHashTable::new();
    let msg = Ft8Message::Standard {
        call_to: "KD9ABC".to_string(),
        call_de: "W9XYZ".to_string(),
        extra: GridField::Grid("FN31".to_string()),
    };

    let payload = pack77(&msg, &mut ht).expect("pack77 ft4");
    let bits: Ft4Bits = payload77_to_ft4bits(&payload);
    let frame = Ft4Codec::encode(&bits);
    let iq = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft4Demod::new(12_000.0, 1_000.0)
        .demodulate(&iq)
        .expect("Ft4Demod failed");
    let bits_out = Ft4Codec::decode_hard(&frame_out).expect("decode_hard failed");
    let decoded = unpack77(&bits_out, &ht);
    assert_eq!(decoded, msg, "Full-stack FT4 Type1 roundtrip mismatch");
}
