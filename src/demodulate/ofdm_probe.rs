// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/ofdm_probe.rs
//
// The COFDM receive probe: the two per-frame quantities an analyzer's
// constellation / decoder display needs, exposed opt-in and at zero cost when
// unused.
//
//   1. The equalizer's output — the complex data-carrier symbols exactly as the
//      demapper saw them (`s_k = r_k / H_k`, after `OfdmEqualizer`, after
//      common-phase-error removal, before `OfdmSoftDemod`). This is where a
//      vector signal analyzer takes its constellation, and it is where
//      `decode_frame_body` already has it: the vector is filled on every frame
//      to measure EVM and then dropped.
//
//   2. A per-coded-bit correction map — for each coded bit, whether the channel
//      corrupted it and whether the inner decoder fixed it.
//
// Both are observations of a decode that happens anyway. Neither changes what
// decodes; see `probing_does_not_change_what_decodes`.

use crate::modulate::ofdm::ConstellationOrder;
use num_complex::Complex32 as C32;
use std::ops::Range;

/// What the channel and the inner decoder each did to one coded bit.
///
/// Derived by comparing three bit-streams in the **coded-bit index space** —
/// what the transmitter sent, what arrived at the demapper, and what the inner
/// decoder's own output re-encodes to:
///
/// | State | arrived correct | decoder agrees | Meaning |
/// | --- | --- | --- | --- |
/// | [`Clean`](Self::Clean) | yes | yes | the channel did not touch it |
/// | [`Corrected`](Self::Corrected) | no | yes | the inner code fixed it |
/// | [`Uncorrected`](Self::Uncorrected) | no | no | arrived wrong, still wrong — the outer code's problem now |
/// | [`Introduced`](Self::Introduced) | yes | no | arrived right, the decoder broke it |
///
/// `Introduced` is not padding for the fourth cell. A belief-propagation
/// decoder that fails to converge flips correct bits, and one that does so *at
/// high SNR* is broken. Having the state means the map can show that; folding
/// it into `Uncorrected` would hide the symptom in with its opposite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BitOutcome {
    /// Arrived correct and the decoder left it correct.
    #[default]
    Clean = 0,
    /// Arrived wrong; the inner decoder fixed it.
    Corrected = 1,
    /// Arrived wrong and the inner decoder did not fix it.
    Uncorrected = 2,
    /// Arrived correct and the inner decoder broke it.
    Introduced = 3,
}

impl BitOutcome {
    /// Classifies one coded bit from the two comparisons that define it.
    #[inline(always)]
    pub(crate) fn classify(arrived_correct: bool, decoder_agrees: bool) -> Self {
        match (arrived_correct, decoder_agrees) {
            (true, true) => BitOutcome::Clean,
            (false, true) => BitOutcome::Corrected,
            (false, false) => BitOutcome::Uncorrected,
            (true, false) => BitOutcome::Introduced,
        }
    }

    /// Whether the channel corrupted this bit — i.e. anything but
    /// [`Clean`](Self::Clean) or [`Introduced`](Self::Introduced).
    ///
    /// Counting these over a frame and dividing by its coded-bit count
    /// reproduces [`channel_ber`](crate::demodulate::OfdmRxFrame::channel_ber)
    /// exactly: the map is that rate's per-bit expansion, not a second
    /// measurement of it.
    ///
    /// **This says nothing about the decoder.** Both states it matches mean
    /// "arrived wrong"; whether the decoder then fixed it is the other half of
    /// the map, and [`decoder_disagreed`](Self::decoder_disagreed) is how to
    /// ask. A consumer checking only this one is blind to the estimate stream
    /// entirely — see that method's note.
    #[inline(always)]
    pub fn arrived_wrong(self) -> bool {
        matches!(self, BitOutcome::Corrected | BitOutcome::Uncorrected)
    }

    /// Whether the inner decoder's output still disagrees with the truth at
    /// this bit — i.e. [`Uncorrected`](Self::Uncorrected) or
    /// [`Introduced`](Self::Introduced).
    ///
    /// The dual of [`arrived_wrong`](Self::arrived_wrong): that one is the
    /// *channel's* half of the map, this is the *decoder's*. Both halves need a
    /// name, because this predicate is the map's whole meaning and a consumer
    /// that spells it out by hand can get it subtly wrong with nothing to say
    /// so.
    ///
    /// The two are independent, not complementary. `Corrected` is
    /// `arrived_wrong && !decoder_disagreed`; `Introduced` is the reverse; a
    /// bit can be both (`Uncorrected`) or neither (`Clean`).
    #[inline(always)]
    pub fn decoder_disagreed(self) -> bool {
        matches!(self, BitOutcome::Uncorrected | BitOutcome::Introduced)
    }
}

/// One frame's probe record: where its symbols and correction map live inside
/// the owning [`OfdmRxProbe`]'s flat buffers, plus the metadata needed to render
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfdmProbeFrame {
    /// The frame's sequence number, or `None` when the link carries no
    /// header-derived one. Always `Some` on the header-bearing COFDM path this
    /// receiver decodes — a frame whose *header* fails never reaches the payload
    /// demapper and so produces no probe record at all.
    pub sequence_num: Option<u32>,
    /// The payload constellation the symbols were demapped against, so a
    /// display can draw the right reference points without re-reading the MCS
    /// table.
    pub constellation: ConstellationOrder,
    /// This frame's span in [`OfdmRxProbe::symbols`].
    ///
    /// **Private, deliberately.** A span is only meaningful against the buffer
    /// it was minted from, and the probe's buffers are cleared and refilled on
    /// every probed call — so a record that outlived its call would index the
    /// wrong frame's data, or past the end. Public spans plus a `Clone` on this
    /// struct made that mistake compile. [`OfdmRxProbe::iter`] hands out
    /// already-resolved slices instead, whose lifetime is tied to the probe, so
    /// the mistake is now unrepresentable and cloning a record is harmless: it
    /// carries metadata and has no way to index anything.
    pub(crate) symbols: Range<usize>,
    /// This frame's span in [`OfdmRxProbe::correction`]. **Empty when
    /// [`decoded`](Self::decoded) is false** — a frame that did not decode has
    /// no ground truth to compare against. Private on the same reasoning as
    /// [`symbols`](Self::symbols).
    pub(crate) correction: Range<usize>,
    /// The inner code's codeword length `n`, so a display can draw codeword
    /// boundaries across the map. `0` when the inner code has no block
    /// structure to draw — [`InnerFec::None`](crate::fec::InnerFec::None) and
    /// the convolutional arm, which terminates once per frame rather than per
    /// codeword.
    pub codeword_bits: usize,
    /// The inner code's information length `k`, on the same terms as
    /// [`codeword_bits`](Self::codeword_bits). For a systematic code the first
    /// `k` bits of each codeword are the message.
    pub codeword_info_bits: usize,
    /// Whether the payload decoded and passed its integrity check. `false` ⇒
    /// there is no ground truth, so [`ProbedFrame::correction`] is empty and
    /// only the symbols are meaningful.
    ///
    /// The map therefore empties exactly when the link is worst. That is
    /// honest — nothing can be measured against a payload that did not
    /// verify — but it has to be rendered as "no ground truth", not as "no
    /// errors".
    pub decoded: bool,
}

/// Reusable per-call diagnostic buffers for
/// [`OfdmFrameStreamDemod::feed_probed`](crate::demodulate::OfdmFrameStreamDemod::feed_probed).
///
/// Cleared and refilled by each probed call; **capacity is retained**, so
/// steady-state probing does not reallocate. That is the reason the caller owns
/// this rather than each frame carrying its own `Option<Vec<_>>`: a probed
/// frame is ~2600 complex symbols and ~5100 outcome bytes, at 8 to 51 frames
/// per second, and `feed` returns *several* frames per call — so a per-frame
/// allocation is paid on every frame and a borrowed buffer cannot live on
/// [`RxFrame`](crate::demodulate::RxFrame) at all.
///
/// # Layout
///
/// [`symbols`](Self::symbols) and [`correction`](Self::correction) are flat
/// across every frame the call produced — read them directly for a bulk view
/// that does not care about frame boundaries (a density accumulator, say).
/// [`iter`](Self::iter) is the per-frame view, and hands out resolved slices
/// rather than spans so a record cannot outlive the call that filled it.
///
/// ```ignore
/// let mut probe = OfdmRxProbe::new();
/// for chunk in stream {
///     // Read the probe after the call that filled it: every probed entry
///     // point clears first, so records do not accumulate across calls.
///     for frame in rx.feed_probed(chunk, &mut probe) { /* ... */ }
///     for f in probe.iter() {
///         plot_constellation(f.symbols, f.meta.constellation);
///         if f.meta.decoded {
///             plot_corrections(f.correction, f.meta.codeword_bits);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct OfdmRxProbe {
    /// Equalized payload symbols, in demap order, for every frame this call
    /// produced.
    pub(crate) symbols: Vec<C32>,
    /// Per-coded-bit outcomes, for every frame this call decoded.
    pub(crate) correction: Vec<BitOutcome>,
    /// Per-frame spans into the two buffers above, plus metadata.
    pub(crate) frames: Vec<OfdmProbeFrame>,
    /// Private scratch, never handed out: the re-encode of the inner decoder's
    /// own output, in the coded-bit domain. Held here so its buffer is reused
    /// across frames — the encode helpers that fill it still allocate their own
    /// intermediates, exactly as
    /// [`with_error_rates`](crate::demodulate::OfdmFrameStreamDemod::with_error_rates)
    /// does today.
    pub(crate) estimate: Vec<u8>,
}

impl OfdmRxProbe {
    /// An empty probe. Reuse one across calls — that is the point of the type.
    pub fn new() -> Self {
        Self::default()
    }

    /// The per-frame records this call produced, in the order the frames were
    /// drained from the buffer.
    pub fn frames(&self) -> &[OfdmProbeFrame] {
        &self.frames
    }

    /// Every frame's equalized payload symbols, flat — a bulk view that does
    /// not care about frame boundaries. Use [`iter`](Self::iter) for the
    /// per-frame view.
    pub fn symbols(&self) -> &[C32] {
        &self.symbols
    }

    /// Every decoded frame's per-coded-bit outcomes, flat, on the same terms
    /// as [`symbols`](Self::symbols).
    pub fn correction(&self) -> &[BitOutcome] {
        &self.correction
    }

    /// Each frame this call produced, with its metadata and both of its slices
    /// already resolved — the way to read a probe.
    ///
    /// The slices borrow the probe, so a [`ProbedFrame`] cannot outlive the
    /// call that filled it: the next `feed_probed` needs `&mut` and the borrow
    /// checker refuses. That is the whole reason this exists rather than a
    /// `symbols_for(&frame)` lookup, which a stale record would silently index
    /// into the wrong frame's data.
    ///
    /// ```ignore
    /// for f in probe.iter() {
    ///     plot_constellation(f.symbols, f.meta.constellation);
    ///     if f.meta.decoded {
    ///         plot_corrections(f.correction, f.meta.codeword_bits);
    ///     }
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = ProbedFrame<'_>> {
        self.frames.iter().map(move |meta| ProbedFrame {
            meta,
            symbols: &self.symbols[meta.symbols.clone()],
            correction: &self.correction[meta.correction.clone()],
        })
    }

    /// Whether this call produced no probe frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Drops the contents, keeping the allocations. Called at the start of each
    /// probed `feed`/`flush`, so a caller never has to.
    pub fn clear(&mut self) {
        self.symbols.clear();
        self.correction.clear();
        self.frames.clear();
    }

    /// Records the current buffer lengths so a frame that fails part-way can be
    /// rolled back to them. See [`rollback`](Self::rollback).
    pub(crate) fn mark(&self) -> ProbeMark {
        ProbeMark {
            symbols: self.symbols.len(),
            correction: self.correction.len(),
            frames: self.frames.len(),
        }
    }

    /// Truncates back to `mark`, discarding whatever a partial frame appended.
    ///
    /// **A probe frame is committed as one unit.** `BodyError::Incomplete`
    /// consumes no buffer and the next `feed` re-runs the frame from its header,
    /// so anything a partial attempt appended must not survive — or the same
    /// frame would be reported twice. `soft_demap` returns `None` on a short
    /// buffer before touching the sink, which makes the common case safe by
    /// construction; the equalized path's second loop is the one that can sink
    /// symbols before its own early return.
    pub(crate) fn rollback(&mut self, mark: ProbeMark) {
        debug_assert!(
            self.symbols.len() >= mark.symbols
                && self.correction.len() >= mark.correction
                && self.frames.len() >= mark.frames,
            "probe buffers must only grow between mark and rollback"
        );
        self.symbols.truncate(mark.symbols);
        self.correction.truncate(mark.correction);
        self.frames.truncate(mark.frames);
    }

    /// Records a frame that reached the demapper but produced no ground truth:
    /// symbols only, an empty correction span, `decoded: false`.
    ///
    /// A failed payload CRC still has a constellation, and the constellation is
    /// precisely where an operator looks when frames stop decoding.
    pub(crate) fn push_undecoded(
        &mut self,
        sym_start: usize,
        constellation: ConstellationOrder,
        sequence_num: Option<u32>,
    ) {
        let end = self.correction.len();
        self.frames.push(OfdmProbeFrame {
            sequence_num,
            constellation,
            symbols: sym_start..self.symbols.len(),
            correction: end..end,
            codeword_bits: 0,
            codeword_info_bits: 0,
            decoded: false,
        });
    }

    /// Records a decoded frame, building its correction map from the three
    /// coded-bit-domain streams that define it: `truth` (the re-encode of the
    /// CRC-verified payload — what the transmitter sent), `received` (the
    /// demapper's hard decisions), and the estimate scratch (the re-encode of
    /// the inner decoder's own output).
    ///
    /// All three are already in the coded-bit index space, ordered
    /// post-inner-interleave and post-`AfterInnerFec` scramble — i.e. exactly
    /// the order the bits were mapped to symbols in, so the map indexes the
    /// same way the symbols do.
    pub(crate) fn push_decoded(
        &mut self,
        sym_start: usize,
        meta: ProbeMeta,
        truth: &[u8],
        received: &[u8],
    ) {
        // Field-level split borrow: the map reads `estimate` while writing
        // `correction`, and both are fields of `self`.
        let Self {
            correction,
            estimate,
            frames,
            symbols,
        } = self;
        let start = correction.len();
        let n = truth.len().min(received.len()).min(estimate.len());
        correction.extend(
            (0..n).map(|i| BitOutcome::classify(received[i] == truth[i], estimate[i] == truth[i])),
        );
        frames.push(OfdmProbeFrame {
            sequence_num: meta.sequence_num,
            constellation: meta.constellation,
            symbols: sym_start..symbols.len(),
            correction: start..correction.len(),
            codeword_bits: meta.codeword_bits,
            codeword_info_bits: meta.codeword_info_bits,
            decoded: true,
        });
    }
}

/// One frame's probe record with both of its slices resolved, as yielded by
/// [`OfdmRxProbe::iter`].
///
/// # Symbols and bits do not index 1:1
///
/// `correction[i]` is coded bit `i`, in the order the bits were mapped to
/// subcarriers; `symbols[j]` is the `j`th data carrier in demap order. So the
/// bits of `symbols[j]` are
///
/// ```text
/// let bps = meta.constellation.bits_per_symbol();
/// &correction[j * bps .. (j + 1) * bps]        // NOT always in range
/// ```
///
/// — but **the symbols carry more bit-slots than the map covers.** A payload is
/// mapped to a whole number of OFDM symbols, so the final one is padded past
/// the end of the coded block. Measured on one test link: 806 QPSK symbols =
/// 1612 slots against a 1536-bit map, 76 slots of padding with no outcome
/// behind them.
///
/// A consumer walking symbols and colouring each by its bits must therefore
/// stop at `correction.len()` rather than run to the end of `symbols`. Walking
/// the map and finding each bit's symbol is always in range and is the safer
/// direction.
#[derive(Debug, Clone, Copy)]
pub struct ProbedFrame<'a> {
    /// This frame's metadata: sequence number, constellation, codeword
    /// geometry, and whether it decoded.
    pub meta: &'a OfdmProbeFrame,
    /// The equalized payload symbols, in demap order.
    pub symbols: &'a [C32],
    /// The per-coded-bit outcomes. **Empty when `meta.decoded` is false** —
    /// that is "no ground truth", not "no errors".
    pub correction: &'a [BitOutcome],
}

/// The per-frame metadata a probe record carries alongside its two spans —
/// everything in [`OfdmProbeFrame`] that is not derived from the buffers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProbeMeta {
    pub(crate) sequence_num: Option<u32>,
    pub(crate) constellation: ConstellationOrder,
    pub(crate) codeword_bits: usize,
    pub(crate) codeword_info_bits: usize,
}

/// The buffer lengths at the start of one frame's probe, for
/// [`OfdmRxProbe::rollback`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProbeMark {
    symbols: usize,
    correction: usize,
    frames: usize,
}
