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
pub use ft4::{Ft4Bits, Ft4Codec};
pub use ft8::{Ft8Bits, Ft8Codec, Ft8DecodeResult, Ft8StreamDecoder};
pub use gray::{gray4_decode, gray4_encode, gray8_decode, gray8_encode};
pub use ldpc::{ldpc_decode_soft, ldpc_encode};
pub use psk31::{Psk31Stream, StreamingViterbi, conv_encode, viterbi_decode, viterbi_decode_hard};
pub use varicode::{
    VARICODE_MAX_BITS, VaricodeDecoder, VaricodeEncoder, varicode_decode, varicode_encode,
};
