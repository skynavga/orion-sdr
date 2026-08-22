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
// left for the super-frame path (`dvb_t_super_frame`).

use super::ofdm_frame::{CodecCache, encode_chain, symbols_for_coded_bits};
use crate::core::Block;
use crate::fec::{CrcKind, InterleaverKind, ScramblerKind, ScramblerPos};
use crate::multicarrier::{CyclicPrefixInsert, IfftBlock, SymbolWindow, TxLowpass};
use crate::waveform::dvb_t::{
    DVB_T_DATA_CARRIERS, DVB_T_FRAME_OUTER, DVB_T_FRAME_OUTER_IL, DVB_T_KMAX, DVB_T_N_FFT,
    DvbTFrameParams, ScatteredPilotMapper, dvb_t_coded_bits_with, dvb_t_frame_fill_with,
    dvb_t_map_symbol, tps_carrier_bins,
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

/// A conformant, preamble-less DVB-T frame modulator. Constructed with the link's
/// transmission parameters ([`DvbTFrameParams`] — guard interval, constellation,
/// code rate, TPS-signalled frame number and cell id); [`modulate`](Self::modulate)
/// produces one frame per call.
#[derive(Debug, Clone)]
pub struct DvbTFrameMod {
    params: DvbTFrameParams,
    /// TX symbol-window roll-off in samples (raised-cosine edge taper). `0`
    /// (default) = no windowing, so the on-air frame is byte-identical.
    window_roll_off: usize,
    /// Optional TX baseband low-pass (spectral mask) over the assembled frame.
    /// `None` (default) leaves the on-air frame byte-identical.
    tx_lowpass: Option<TxLowpass>,
}

impl DvbTFrameMod {
    /// Builds a modulator for a link with the given transmission parameters.
    pub fn new(params: DvbTFrameParams) -> Self {
        Self {
            params,
            window_roll_off: 0,
            tx_lowpass: None,
        }
    }

    /// Enables TX symbol windowing with a `roll_off`-sample raised-cosine taper
    /// at each symbol edge, reducing out-of-band emission. `0` (the default)
    /// disables it. DVB-T frames are preamble-less — every symbol is CP-bearing —
    /// so every symbol is windowed. The taper is only RX-transparent when the
    /// receiver's window back-off is paired to it (`roll_off ≤ cp_len/2` with
    /// back-off `cp_len/2`); see
    /// [`DvbTFrameDemod::with_rx_window_backoff`](crate::demodulate::DvbTFrameDemod::with_rx_window_backoff).
    /// The continual/scattered/TPS pilots are unaffected — windowing touches only
    /// the time-domain guard samples, not the subcarrier allocation.
    pub fn with_symbol_window(mut self, roll_off: usize) -> Self {
        self.window_roll_off = roll_off;
        self
    }

    /// Enables a TX baseband low-pass (spectral mask) across the assembled
    /// frame, applied after any symbol taper. `None`/absent (the default) leaves
    /// the frame byte-identical.
    ///
    /// This is the DVB-T lever that **exceeds** the symbol-windowing ceiling: it
    /// attenuates out-of-band energy directly in the frequency domain, and DVB-T
    /// comes with room to do it in — 1705 of 2048 bins are active, so there is a
    /// real null band for the transition. It changes nothing about how a
    /// receiver decodes (the scattered-pilot equalizer absorbs the filter like
    /// any other channel), but its group delay must land in guard the receiver
    /// discards: pair it with
    /// [`DvbTFrameDemod::with_rx_window_backoff`](crate::demodulate::DvbTFrameDemod::with_rx_window_backoff)
    /// and keep `roll_off + group_delay ≤ min(cp_len − b, b)`
    /// ([`TxLowpass::fits_guard`]). A **long guard** buys a sharper mask: G1/4
    /// (`cp_len = 512`) affords eight times the filter length of G1/32.
    ///
    /// [`TxLowpass::for_null_band`] placed against DVB-T's own band edge is
    /// [`for_dvb_t_2k`](Self::tx_lowpass_for_2k).
    pub fn with_tx_lowpass(mut self, lowpass: TxLowpass) -> Self {
        self.tx_lowpass = Some(lowpass);
        self
    }

    /// A spectral mask sized for the fixed DVB-T 2K band edge (active carriers
    /// `±852` of 2048), leaving `num_taps` and `stopband_db` to the caller —
    /// `num_taps` is what the guard budget constrains.
    /// [`TxLowpass::taps_for_null_band`] with the same arguments suggests a
    /// length.
    pub fn tx_lowpass_for_2k(num_taps: usize, stopband_db: f32) -> TxLowpass {
        TxLowpass::for_null_band(DVB_T_N_FFT, DVB_T_KMAX / 2, num_taps, stopband_db)
    }

    /// The transmission parameters this modulator was built with.
    pub fn params(&self) -> DvbTFrameParams {
        self.params
    }

    /// Modulates `payload` (the MPEG-TS payload bytes) into one conformant,
    /// preamble-less DVB-T frame: TS packetization + energy dispersal, the DVB-T
    /// payload FEC, Figure-9a mapping through the four-phase scattered-pilot grid,
    /// and the TPS word DBPSK-woven onto the 17 TPS carriers across the symbols.
    /// The frame spans `max(payload symbols, 68)` OFDM symbols so a full TPS block
    /// is present.
    ///
    /// A short payload that does not fill the frame is **stuffed with MPEG-2 null
    /// packets** (PID `0x1FFF`) — §4.4 ("all symbols contain data") and §4.3.1
    /// (randomization stays active with no program input): a compliant DVB-T
    /// signal never leaves data carriers zeroed. The RX trims the recovered
    /// payload back to `payload_len`, so the stuffing is transparent.
    ///
    /// Stuffing stops at the largest packet count whose coded stream still
    /// **fits** ([`dvb_t_frame_fill`](crate::waveform::dvb_t::dvb_t_frame_fill)),
    /// and the carriers past it repeat the coded stream's head. Nothing is
    /// truncated, so a receiver reconstructing what was sent — the gate behind
    /// [`DvbTFrameDemod::with_error_rates`](crate::demodulate::DvbTFrameDemod::with_error_rates)
    /// — never asks its decoder for bits that never went on air. (Exact-fit with
    /// no stuffing at all is a super-frame property — §4.7, Table 16 — handled by
    /// the super-frame path, not here.)
    pub fn modulate(&self, payload: &[u8]) -> DvbTFrame {
        let params = self.params;
        let cache = CodecCache::new();
        let base = params.config();
        let cp_len = base.carrier_plan.cp_len();
        let n_fft = DVB_T_N_FFT;
        let sps = n_fft + cp_len;
        let vbits = params.constellation().bits_per_symbol();
        let bits_per_sym = DVB_T_DATA_CARRIERS * vbits;

        // 1. TS-packetize the real payload; decide the frame's symbol count from
        //    it, padded to a full 68-symbol TPS block.
        let mut ts = ts_packetize(payload);
        let n_real_packets = ts.len() / TS_PACKET_LEN;
        let payload_syms = symbols_for_coded_bits(
            &base,
            params.constellation(),
            dvb_t_coded_bits_with(params, n_real_packets, &cache),
        );
        let n_symbols = payload_syms.max(TPS_SYMBOLS_PER_FRAME);

        // 2. Stuff null packets up to the largest count whose coded stream still
        //    FITS the frame's data carriers (the shared rule — see
        //    `dvb_t_frame_fill`), then apply energy dispersal to the whole
        //    (payload + null) stream.
        let fill = dvb_t_frame_fill_with(params, n_real_packets.max(1), n_symbols, &cache);
        // `n_symbols` was derived from the payload's own coded length, so the
        // payload always fits and "largest that fits" never drops a real packet.
        debug_assert!(
            fill.n_ts_packets >= n_real_packets,
            "frame filling must not drop real payload packets"
        );
        debug_assert_eq!(fill.capacity_bits, n_symbols * bits_per_sym);
        ts_stuff_null_packets(&mut ts, fill.n_ts_packets);
        ts_energy_disperse(&mut ts);

        // 3. Payload FEC: RS(204,188) + K=7 conv + Forney interleaver. No extra
        //    scrambler here — energy dispersal was applied at the TS layer.
        let mut coded_bits = encode_chain(
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
        debug_assert_eq!(coded_bits.len(), fill.coded_bits);

        // 3b. Fill the carriers the coded stream leaves over by repeating its
        //     head. §4.4 wants every data carrier modulated, and the per-packet
        //     coded step and the frame's capacity coincide at no packet count, so
        //     something has to cover the remainder — but nothing may be
        //     truncated, because a receiver measuring the link re-encodes what it
        //     recovered and would then be comparing against bits that were never
        //     sent.
        //
        //     A REPEAT rather than zeros, because energy dispersal is applied at
        //     the TS layer ahead of the FEC: the coded stream is already
        //     whitened, so repeating any of it stays whitened. Zeros would not —
        //     at QPSK r1/2 the remainder runs past 1512 bits, so a zero fill
        //     would put an entire OFDM symbol on one constellation point.
        //
        //     These bits are never decoded: the receiver's block plan ends at
        //     `fill.coded_bits`, which is where the repeat begins.
        let filler = fill.filler_bits();
        debug_assert!(
            filler < coded_bits.len(),
            "the remainder is under one packet's coded step, far below a frame"
        );
        coded_bits.extend_from_within(..filler);
        debug_assert_eq!(coded_bits.len(), fill.capacity_bits);

        // 4. Map symbols: data via the scattered grid + DVB-T constellation, then
        //    overwrite the TPS carriers with the DBPSK cells for that symbol.
        let mut mapper = ScatteredPilotMapper::new(params.guard());
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
                // `coded_bits` is exactly `capacity_bits` long after the filler,
                // so every carrier has bits and none can be left zeroed — no
                // bounds arm here, and none reachable.
                let bit_base = s * bits_per_sym + c * vbits;
                *slot =
                    dvb_t_map_symbol(&coded_bits[bit_base..bit_base + vbits]).expect("DVB-T order");
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

        // Optional TX symbol windowing. DVB-T is preamble-less, so every symbol
        // is a CP-bearing OFDM symbol and every one is tapered; the taper touches
        // only guard samples, leaving the continual/scattered/TPS pilots intact.
        if self.window_roll_off > 0 {
            let mut win = SymbolWindow::new(sps, self.window_roll_off);
            for s in 0..n_symbols {
                let symbol: Vec<C32> = iq[s * sps..(s + 1) * sps].to_vec();
                win.process(&symbol, &mut iq[s * sps..(s + 1) * sps]);
            }
        }

        // Optional TX baseband low-pass, last and across the whole frame: it is
        // a spectral filter spanning symbol boundaries, not a per-symbol taper.
        // Same-length and group-delay-compensated, so the symbol grid a
        // guard-interval receiver acquires is unmoved.
        if let Some(lowpass) = self.tx_lowpass {
            lowpass.apply(&mut iq);
        }

        DvbTFrame {
            iq,
            n_symbols,
            samples_per_symbol: sps,
        }
    }
}
