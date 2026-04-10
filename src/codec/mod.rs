// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod crc;
pub mod ft4;
pub mod ft8;
pub mod gray;
pub mod ldpc;
pub mod psk31;
pub mod varicode;

pub use crc::ft8_crc14;
pub use ft4::{Ft4Codec, Ft4Bits};
pub use ft8::{Ft8Codec, Ft8Bits, Ft8StreamDecoder, Ft8DecodeResult};
pub use gray::{gray8_encode, gray8_decode, gray4_encode, gray4_decode};
pub use ldpc::{ldpc_encode, ldpc_decode_soft};
pub use psk31::{conv_encode, viterbi_decode, viterbi_decode_hard, StreamingViterbi, Psk31Stream};
pub use varicode::{
    varicode_encode, varicode_decode,
    VaricodeEncoder, VaricodeDecoder,
    VARICODE_MAX_BITS,
};
