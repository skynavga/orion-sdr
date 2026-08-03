// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/dvb_t_frame.rs
//
// The conformant DVB-T on-air frame MODULATOR (ETSI EN 300 744): a preamble-less
// OFDM frame carrying an MPEG-2 transport-stream payload, TPS signalling on the
// 17 reserved carriers, and scattered/continual pilots. Unlike the generic
// `OfdmFrameMod` (which prepends a Schmidl & Cox preamble + an `OrionSdr`
// header), this emits exactly what a real DVB-T receiver expects — a stream of
// OFDM symbols with the transmission parameters carried by TPS rather than a
// prepended header. The `demodulate::dvb_t_frame` module is its exact inverse.
//
// This is a per-standard ORCHESTRATOR over the crate's shared, separately-
// callable pipeline stages — it introduces no new generic abstraction:
//   • payload FEC     — `encode_chain` (RS(204,188) + K=7 conv + Forney I=12);
//   • energy dispersal— `waveform::dvb_t_ts` (188-byte TS packets);
//   • constellation   — `dvb_t_map_symbol` (Figure-9a);
//   • pilots/grid     — `ScatteredPilotMapper` (Phase 2);
//   • TPS             — `dvb_t_tps::{TpsWord, TpsEncoder}`.
// A future coherent-OFDM standard would get its own thin orchestrator over the
// same stages.
//
// SCOPE. One OFDM frame is 68 symbols; a small payload occupies fewer, so a
// single frame is padded to at least 68 symbols so a full TPS block is present.
// The caller tracks the payload length (real DVB-T length is implicit in the
// continuous TS). Multi-frame streaming and super-frame sync-word alternation are
// left for a future streaming path.

use super::ofdm_frame::{CodecCache, block_plan, encode_chain, symbols_for_coded_bits};
use crate::core::Block;
use crate::fec::{CrcKind, InterleaverKind, ScramblerKind, ScramblerPos};
use crate::multicarrier::{CyclicPrefixInsert, IfftBlock};
use crate::waveform::dvb_t::{
    DVB_T_DATA_CARRIERS, DVB_T_FRAME_OUTER, DVB_T_FRAME_OUTER_IL, DVB_T_N_FFT, DvbTFrameParams,
    ScatteredPilotMapper, dvb_t_map_symbol, tps_carrier_bins,
};
use crate::waveform::dvb_t_tps::{TPS_SYMBOLS_PER_FRAME, TpsEncoder};
use crate::waveform::dvb_t_ts::{
    TS_PACKET_LEN, ts_energy_disperse, ts_packetize, ts_stuff_null_packets,
};
use num_complex::Complex32 as C32;

/// A modulated DVB-T frame: the time-domain IQ plus the numerology a receiver
/// needs to acquire it (all also recoverable from the signal; returned for
/// caller/test convenience).
#[derive(Debug, Clone)]
pub struct DvbTFrame {
    /// Time-domain baseband IQ (no preamble; a whole number of OFDM symbols).
    pub iq: Vec<C32>,
    /// Number of OFDM symbols in the frame.
    pub n_symbols: usize,
    /// Samples per OFDM symbol (`n_fft + cp_len`).
    pub samples_per_symbol: usize,
}

/// Modulates `payload` into one conformant, preamble-less DVB-T frame: TS
/// packetization + energy dispersal, the DVB-T payload FEC, Figure-9a mapping
/// through the four-phase scattered-pilot grid, and the TPS word DBPSK-woven onto
/// the 17 TPS carriers across the symbols. `payload` is the TS payload (packetized
/// here). The frame spans `max(payload symbols, 68)` OFDM symbols so a full TPS
/// block is present.
///
/// A short payload that does not fill the frame is **stuffed with MPEG-2 null
/// packets** (PID `0x1FFF`) so the coded stream reaches every data carrier —
/// §4.4 ("all symbols contain data") and §4.3.1 (randomization stays active with
/// no program input): a compliant DVB-T signal never leaves data carriers zeroed.
/// The RX trims the recovered payload back to `payload_len`, so the stuffing is
/// transparent. (Exact-fit with no stuffing is a super-frame property — §4.7,
/// Table 16 — handled by the future super-frame path, not here.)
pub fn dvb_t_frame_modulate(params: DvbTFrameParams, payload: &[u8]) -> DvbTFrame {
    let cache = CodecCache::new();
    let base = params.config();
    let cp_len = base.carrier_plan.cp_len();
    let n_fft = DVB_T_N_FFT;
    let sps = n_fft + cp_len;
    let vbits = params.constellation.bits_per_symbol();
    let bits_per_sym = DVB_T_DATA_CARRIERS * vbits;

    // Coded-bit count the FEC chain produces from a whole number of TS packets
    // (`n_pkt` × 188-byte info), via the shared block plan.
    let coded_bits_for_packets = |n_pkt: usize| -> usize {
        block_plan(
            n_pkt * TS_PACKET_LEN,
            CrcKind::None,
            DVB_T_FRAME_OUTER,
            params.inner(),
            DVB_T_FRAME_OUTER_IL,
            InterleaverKind::None,
            &cache,
        )
        .coded_bits
    };

    // 1. TS-packetize the real payload; decide the frame's symbol count from it,
    //    padded to a full 68-symbol TPS block.
    let mut ts = ts_packetize(payload);
    let n_real_packets = ts.len() / TS_PACKET_LEN;
    let payload_syms = symbols_for_coded_bits(
        &base,
        params.constellation,
        coded_bits_for_packets(n_real_packets),
    );
    let n_symbols = payload_syms.max(TPS_SYMBOLS_PER_FRAME);

    // 2. Stuff null packets until the coded stream fills the frame's data
    //    carriers (so no carrier is left zeroed), then apply energy dispersal to
    //    the whole (payload + null) stream.
    let capacity_bits = n_symbols * bits_per_sym;
    let mut target_packets = n_real_packets.max(1);
    while coded_bits_for_packets(target_packets) < capacity_bits {
        target_packets += 1;
    }
    ts_stuff_null_packets(&mut ts, target_packets);
    ts_energy_disperse(&mut ts);

    // 3. Payload FEC: RS(204,188) + K=7 conv + Forney interleaver. No extra
    //    scrambler here — energy dispersal was applied at the TS layer.
    let coded_bits = encode_chain(
        &ts,
        CrcKind::None, // RS(204,188) is the payload protection; TS carries its own framing
        DVB_T_FRAME_OUTER,
        params.inner(),
        DVB_T_FRAME_OUTER_IL,
        InterleaverKind::None,
        ScramblerKind::None,
        ScramblerPos::BeforeOuterFec,
        0,
        &cache,
    );
    debug_assert!(
        coded_bits.len() >= capacity_bits,
        "null-packet stuffing must fill every data carrier"
    );

    // 4. Map symbols: data via the scattered grid + DVB-T constellation, then
    //    overwrite the TPS carriers with the DBPSK cells for that symbol.
    let mut mapper = ScatteredPilotMapper::new(params.guard);
    let mut tps_enc = TpsEncoder::new();
    let tps_block = params.tps_word().pack();
    let tps_bins = tps_carrier_bins();

    let mut ifft = IfftBlock::new(n_fft);
    let mut cp_insert = CyclicPrefixInsert::new(n_fft, cp_len);
    let mut data_syms = vec![C32::default(); DVB_T_DATA_CARRIERS];
    let mut freq = vec![C32::default(); n_fft];
    let mut time = vec![C32::default(); n_fft];
    let mut iq = vec![C32::default(); n_symbols * sps];

    for s in 0..n_symbols {
        for (c, slot) in data_syms.iter_mut().enumerate() {
            // Stuffing guarantees the coded stream reaches every carrier; a short
            // final overrun (coded bits past the last carrier) is simply unused,
            // as in a continuous transmission crossing a frame boundary.
            let bit_base = s * bits_per_sym + c * vbits;
            *slot = if bit_base + vbits <= coded_bits.len() {
                dvb_t_map_symbol(&coded_bits[bit_base..bit_base + vbits]).expect("DVB-T order")
            } else {
                C32::default()
            };
        }
        mapper.map_symbol(&data_syms, &mut freq);
        let tps_bit = tps_block[s % TPS_SYMBOLS_PER_FRAME];
        let cells = tps_enc.next_symbol(tps_bit);
        for (&bin, &cell) in tps_bins.iter().zip(cells.iter()) {
            freq[bin] = cell;
        }
        if (s + 1) % TPS_SYMBOLS_PER_FRAME == 0 {
            tps_enc.reset();
        }
        ifft.process(&freq, &mut time);
        cp_insert.process(&time, &mut iq[s * sps..(s + 1) * sps]);
    }

    DvbTFrame {
        iq,
        n_symbols,
        samples_per_symbol: sps,
    }
}
