// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/dvb_t_probe.rs
//
// The DVB-T receive probe: the DVB-T counterpart of `ofdm_probe`, exposing the
// two per-frame quantities an analyzer's constellation / decoder display needs,
// opt-in and at zero cost when unused.
//
//   1. The equalizer's output — the 1512 complex data-carrier symbols per OFDM
//      symbol exactly as the demapper saw them (after `OfdmEqualizer` against the
//      scattered/continual pilots, before `dvb_t_soft_llr`). This is where a
//      vector signal analyzer takes its constellation.
//
//   2. A per-coded-bit correction map — for each coded bit, whether the channel
//      corrupted it and whether the inner Viterbi decoder fixed it.
//
// Both are observations of a decode that happens anyway; neither changes what
// decodes. The generic path's `BitOutcome` is reused rather than duplicated: it
// is defined over coded bits and a re-encode comparison, with no COFDM-specific
// assumption in it.
//
// The DVB-T shape differs from the generic one in two ways that are deliberate
// rather than incidental, and both are argued at the type that carries them:
// there is no sequence number (DVB-T has no frame header — see
// `DvbTProbeFrame::tps`), and there is no codeword geometry (its inner code is
// convolutional — see `DvbTProbeFrame`).

use crate::demodulate::ofdm_probe::BitOutcome;
use crate::modulate::ofdm::ConstellationOrder;
use crate::waveform::dvb_t_tps::TpsWord;
use num_complex::Complex32 as C32;
use std::ops::Range;

/// One DVB-T frame's probe record: where its symbols and correction map live
/// inside the owning [`DvbTRxProbe`]'s flat buffers, plus the metadata needed to
/// render them.
///
/// # No codeword geometry
///
/// [`OfdmProbeFrame`](crate::demodulate::OfdmProbeFrame) carries `codeword_bits`
/// / `codeword_info_bits` so a display can draw codeword boundaries across the
/// map. DVB-T's inner code is always `ConvCode::DvbK7`, a convolutional code
/// that terminates once per frame and has no block structure to draw — the
/// generic path already reports `(0, 0)` for that arm. Carrying a pair of
/// permanent zeroes would invite a consumer to divide by them; the fields are
/// omitted instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvbTProbeFrame {
    /// The TPS word this frame signalled.
    ///
    /// **This is where a sequence number would go, and DVB-T has none.** The
    /// generic path's `sequence_num` comes from a frame header that DVB-T does
    /// not transmit; what it has instead is a TPS `frame_number` in `0..=3`,
    /// which wraps every super-frame. Synthesising a monotonic counter from it
    /// would be a lie a consumer could not detect — gap arithmetic across a wrap
    /// would silently report four-frame jumps that never happened — so the whole
    /// word is carried and the frame number left as what it is.
    pub tps: TpsWord,
    /// The constellation the symbols were demapped against, so a display can
    /// draw the right reference points.
    pub constellation: ConstellationOrder,
    /// This frame's span in [`DvbTRxProbe::symbols`].
    ///
    /// **Private, deliberately** — the same reasoning as
    /// [`OfdmProbeFrame::symbols`](crate::demodulate::OfdmProbeFrame). A span is
    /// only meaningful against the buffer it was minted from, and the probe's
    /// buffers are cleared and refilled on every probed call, so a record that
    /// outlived its call would index the wrong frame's data or past the end.
    /// [`DvbTRxProbe::iter`] hands out already-resolved slices instead.
    pub(crate) symbols: Range<usize>,
    /// This frame's span in [`DvbTRxProbe::correction`]. **Empty when
    /// [`decoded`](Self::decoded) is false.** Private on the same reasoning as
    /// [`symbols`](Self::symbols).
    pub(crate) correction: Range<usize>,
    /// Whether the payload decoded and passed its outer Reed–Solomon check
    /// (DVB-T carries no CRC, so RS is the integrity check). `false` ⇒ there is
    /// no ground truth, so the correction map is empty and only the symbols are
    /// meaningful.
    ///
    /// The map therefore empties exactly when the link is worst. That is honest
    /// — nothing can be measured against a payload that did not verify — but it
    /// has to be rendered as "no ground truth", not as "no errors".
    pub decoded: bool,
}

/// Reusable per-call diagnostic buffers for
/// [`DvbTFrameStreamDemod::feed_probed`](crate::demodulate::DvbTFrameStreamDemod::feed_probed).
///
/// Cleared and refilled by each probed call; **capacity is retained**, so
/// steady-state probing does not reallocate. That is why the caller owns this
/// rather than each frame carrying its own `Option<Vec<_>>`: a DVB-T frame is
/// 1512 × 68 = 102,816 complex symbols (~823 KB) plus its correction map, and
/// `feed` can return several frames per call — a per-frame allocation would be
/// paid on every frame of a continuous broadcast stream.
///
/// # Layout
///
/// [`symbols`](Self::symbols) and [`correction`](Self::correction) are flat
/// across every frame the call produced — read them directly for a bulk view
/// that does not care about frame boundaries. [`iter`](Self::iter) is the
/// per-frame view, and hands out resolved slices rather than spans so a record
/// cannot outlive the call that filled it.
///
/// ```ignore
/// let mut probe = DvbTRxProbe::new();
/// for chunk in stream {
///     // Read the probe after the call that filled it: every probed entry
///     // point clears first, so records do not accumulate across calls.
///     for frame in rx.feed_probed(chunk, &mut probe) { /* ... */ }
///     for f in probe.iter() {
///         plot_constellation(f.symbols, f.meta.constellation);
///         if f.meta.decoded {
///             plot_corrections(f.correction);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct DvbTRxProbe {
    /// Equalized data-carrier symbols, in demap order, for every frame this
    /// call produced.
    pub(crate) symbols: Vec<C32>,
    /// Per-coded-bit outcomes, for every frame this call decoded.
    pub(crate) correction: Vec<BitOutcome>,
    /// Per-frame spans into the two buffers above, plus metadata.
    pub(crate) frames: Vec<DvbTProbeFrame>,
    /// Private scratch, never handed out: the re-encode of the inner decoder's
    /// own output, in the coded-bit domain. Held here so its buffer is reused
    /// across frames.
    pub(crate) estimate: Vec<u8>,
    /// Private scratch: the demapper's hard decisions, one per coded bit. The
    /// unprobed decode never materializes these — it compares LLRs against the
    /// re-encode in place — but the correction map is indexed per bit and needs
    /// them to exist. Held here for the same reuse reason as `estimate`.
    pub(crate) hard: Vec<u8>,
}

impl DvbTRxProbe {
    /// An empty probe. Reuse one across calls — that is the point of the type.
    pub fn new() -> Self {
        Self::default()
    }

    /// The per-frame records this call produced, in the order the frames were
    /// drained from the buffer.
    pub fn frames(&self) -> &[DvbTProbeFrame] {
        &self.frames
    }

    /// Every frame's equalized data-carrier symbols, flat — a bulk view that
    /// does not care about frame boundaries. Use [`iter`](Self::iter) for the
    /// per-frame view.
    pub fn symbols(&self) -> &[C32] {
        &self.symbols
    }

    /// Every decoded frame's per-coded-bit outcomes, flat, on the same terms as
    /// [`symbols`](Self::symbols).
    pub fn correction(&self) -> &[BitOutcome] {
        &self.correction
    }

    /// Each frame this call produced, with its metadata and both of its slices
    /// already resolved — the way to read a probe.
    ///
    /// The slices borrow the probe, so a [`DvbTProbedFrame`] cannot outlive the
    /// call that filled it: the next `feed_probed` needs `&mut` and the borrow
    /// checker refuses.
    pub fn iter(&self) -> impl Iterator<Item = DvbTProbedFrame<'_>> {
        self.frames.iter().map(move |meta| DvbTProbedFrame {
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
    pub(crate) fn mark(&self) -> DvbTProbeMark {
        DvbTProbeMark {
            symbols: self.symbols.len(),
            correction: self.correction.len(),
            frames: self.frames.len(),
        }
    }

    /// Whether a frame record was committed since `mark`.
    ///
    /// The test a failed decode is rolled back by. A frame that reached the
    /// demapper but did not verify **commits** a record (symbols, empty map,
    /// `decoded: false`) and still returns an error, so "the decode failed" is
    /// not on its own a reason to discard what was appended — this distinguishes
    /// that case from one that failed before committing anything.
    pub(crate) fn committed_since(&self, mark: DvbTProbeMark) -> bool {
        self.frames.len() > mark.frames
    }

    /// Truncates back to `mark`, discarding whatever a partial frame appended.
    ///
    /// **A probe frame is committed as one unit.** A DVB-T frame appends its
    /// symbols as it demaps them, symbol by symbol, and any of the decode steps
    /// after that can still fail — a TPS word that will not resolve, an
    /// uncorrectable RS block. The stream receiver consumes past a failed frame
    /// and moves on, so a partial attempt's symbols must not survive to be
    /// attributed to the next one.
    pub(crate) fn rollback(&mut self, mark: DvbTProbeMark) {
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
    /// A frame whose Reed–Solomon stage failed still has a constellation, and
    /// the constellation is precisely where an operator looks when frames stop
    /// decoding.
    pub(crate) fn push_undecoded(
        &mut self,
        sym_start: usize,
        constellation: ConstellationOrder,
        tps: TpsWord,
    ) {
        let end = self.correction.len();
        self.frames.push(DvbTProbeFrame {
            tps,
            constellation,
            symbols: sym_start..self.symbols.len(),
            correction: end..end,
            decoded: false,
        });
    }

    /// Records a decoded frame, building its correction map from the three
    /// coded-bit-domain streams that define it: `truth` (the re-encode of the
    /// RS-verified TS — what the transmitter sent), the hard-decision scratch
    /// (what arrived at the demapper), and the estimate scratch (the re-encode
    /// of the inner decoder's own output).
    ///
    /// All three are in the same coded-bit index space — the order the bits were
    /// mapped to subcarriers — so the map indexes the same way the symbols do.
    pub(crate) fn push_decoded(
        &mut self,
        sym_start: usize,
        constellation: ConstellationOrder,
        tps: TpsWord,
        truth: &[u8],
    ) {
        // Field-level split borrow: the map reads `estimate`/`hard` while
        // writing `correction`, and all are fields of `self`.
        let Self {
            correction,
            estimate,
            hard,
            frames,
            symbols,
        } = self;
        let start = correction.len();
        let n = truth.len().min(hard.len()).min(estimate.len());
        correction.extend(
            (0..n).map(|i| BitOutcome::classify(hard[i] == truth[i], estimate[i] == truth[i])),
        );
        frames.push(DvbTProbeFrame {
            tps,
            constellation,
            symbols: sym_start..symbols.len(),
            correction: start..correction.len(),
            decoded: true,
        });
    }
}

/// One DVB-T frame's probe record with both of its slices resolved, as yielded
/// by [`DvbTRxProbe::iter`].
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
/// — but **the symbols carry more bit-slots than the map covers.** A DVB-T frame
/// is a whole number of OFDM symbols (68 of them), each carrying exactly 1512
/// data cells, and the coded payload does not in general fill that grid to the
/// last bit; the remainder is padding with no outcome behind it.
///
/// A consumer walking symbols and colouring each by its bits must therefore stop
/// at `correction.len()` rather than run to the end of `symbols`. Walking the map
/// and finding each bit's symbol is always in range and is the safer direction.
#[derive(Debug, Clone, Copy)]
pub struct DvbTProbedFrame<'a> {
    /// This frame's metadata: TPS word, constellation, and whether it decoded.
    pub meta: &'a DvbTProbeFrame,
    /// The equalized data-carrier symbols, in demap order.
    pub symbols: &'a [C32],
    /// The per-coded-bit outcomes. **Empty when `meta.decoded` is false** —
    /// that is "no ground truth", not "no errors".
    pub correction: &'a [BitOutcome],
}

/// The buffer lengths at the start of one frame's probe, for
/// [`DvbTRxProbe::rollback`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct DvbTProbeMark {
    symbols: usize,
    correction: usize,
    frames: usize,
}
