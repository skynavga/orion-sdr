// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/tx_lowpass.rs
use crate::dsp::fir::{FirLowpassIq, kaiser_num_taps, kaiser_transition_norm};
use num_complex::Complex32 as C32;

/// TX baseband low-pass (spectral-mask) filter for an assembled multicarrier
/// stream: the one out-of-band lever that is **not** bounded by the symbol
/// windowing ceiling.
///
/// Symbol windowing ([`SymbolWindow`](super::SymbolWindow)) attacks the skirt
/// indirectly, by softening the symbol-boundary discontinuity; its taper is
/// capped at `cp_len/2`, which caps the payoff (~11 dB measured). This filter
/// attacks the skirt directly in the frequency domain — a low-pass across the
/// **composite** stream, spanning symbol boundaries — so its attenuation stacks
/// on top of windowing instead of sharing the same budget.
///
/// # What the receiver sees
///
/// The filter is applied with [`FirLowpassIq::filter_aligned`], which is
/// same-length and group-delay-compensated, so stream length and symbol
/// boundaries are unchanged and a fixed-`sps` strided receiver is unaffected.
/// What the receiver then sees is a **linear channel it already inverts**: for
/// an FFT window at back-off `b`, each windowed sample combines symbol samples
/// spanning `±d` around the window, where `d` is the filter's group delay. When
/// that reach stays inside the symbol (and clear of any symbol-window taper),
/// the cyclic prefix makes the combination an exact *circular* convolution, so
/// the FFT sees each subcarrier scaled by a single complex `H[k]` — which the
/// pilot/training channel estimate divides out like any other channel. Nothing
/// about the demodulator's decoding needs to change.
///
/// # Guard budget — the filter needs the back-off too
///
/// That argument holds only while the filter's reach and any taper both fall in
/// the guard samples the receiver discards:
///
/// ```text
/// roll_off + group_delay ≤ min(cp_len − backoff, backoff)
/// ```
///
/// maximized, as for windowing, at `backoff = cp_len/2`. See
/// [`fits_guard`](Self::fits_guard). Note what this means at `backoff = 0`: the
/// slack is zero, so **a group-delay-compensated filter is not transparent
/// against a receiver pinned at the cyclic-prefix boundary** — centring the
/// response makes half of it a pre-echo the standard window has no room for.
/// The RX FFT-window back-off is therefore the shared enabler for *both* TX
/// shaping levers, not a windowing-only requirement: set `backoff ≈ cp_len/2`
/// and spend the resulting slack on `roll_off + group_delay`.
///
/// Overrunning the budget does not corrupt the decode abruptly — it leaks a
/// little inter-symbol energy the equalizer cannot invert — but it is the
/// budget to design against, and it is why a **long guard** (DVB-T G1/4 gives
/// `cp_len = 512`) buys a much sharper mask than a short one.
///
/// A bare, *unequalized* demod sees the filter's in-band tilt uncorrected —
/// the same caveat that applies to the RX window back-off.
///
/// # It needs somewhere to filter
///
/// A mask can only attenuate bandwidth the signal does not occupy. A COFDM plan
/// that fills every bin out to Nyquist leaves no null band for a transition, so
/// pair this with an edge-carrier guard
/// ([`CarrierPlan::with_contiguous_data`](super::CarrierPlan::with_contiguous_data)):
/// the guard makes the room, the mask uses it. DVB-T comes with the room built
/// in (1705 of 2048 bins are active).
///
/// # Acquisition budget (preamble-bearing waveforms)
///
/// The filter is applied to the whole assembled burst, preamble included — a
/// real transmitter band-limits everything it emits, and filtering only part of
/// a burst would reintroduce the spectral step. A Schmidl & Cox preamble is
/// `num_repeats` copies of a `repeat_len` segment, and repetition survives
/// filtering wherever the taps see only repeated samples: exactly for outputs
/// at least `group_delay` from each end of the repeated region. So the second
/// sizing rule, alongside the guard budget, is **`group_delay ≪ repeat_len`**.
/// Preamble-less waveforms (DVB-T, which acquires from the cyclic prefix) are
/// bound only by the guard budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TxLowpass {
    /// −6 dB cutoff as a fraction of `fs` (`0 < cutoff_norm < 0.5`). Applied to
    /// a complex baseband stream the filter passes `±cutoff_norm · fs`.
    pub cutoff_norm: f32,
    /// FIR length in taps. Forced odd and ≥ 3 at design time, giving an integer
    /// group delay of `(num_taps − 1)/2`. **The caller owns this number**: it is
    /// the quantity the cyclic-prefix budget constrains.
    pub num_taps: usize,
    /// Kaiser stop-band attenuation target in dB (≥ 21; below that the window
    /// degenerates to rectangular). With `num_taps` fixed, this trades
    /// stop-band depth against transition width.
    pub stopband_db: f32,
}

impl TxLowpass {
    /// A filter with an explicit cutoff, length, and stop-band target.
    pub fn new(cutoff_norm: f32, num_taps: usize, stopband_db: f32) -> Self {
        Self {
            cutoff_norm,
            num_taps,
            stopband_db,
        }
    }

    /// Places the cutoff for a carrier layout whose outermost occupied
    /// subcarrier is `occupied_half` bins from DC in an `n_fft` grid, so that
    /// the **pass band ends at that outermost carrier** and the stop band is
    /// reached as early inside the null band as the length allows.
    ///
    /// Pushing the transition hard against the band edge — rather than centring
    /// it in the null band — is what makes the filter a *mask*: every bin past
    /// the transition gets the full `stopband_db`, and out-of-band energy is
    /// dominated by the bins nearest the edge. The cutoff is clamped so the stop
    /// band still ends by Nyquist; if `num_taps` is too short for the transition
    /// to fit at all (see [`transition_fits`](Self::transition_fits)) it falls
    /// back to centring, which is the best a filter that short can do.
    ///
    /// `num_taps` stays the caller's choice because it, not the cutoff, is what
    /// the guard budget constrains — see [`taps_for_null_band`](Self::taps_for_null_band).
    pub fn for_null_band(
        n_fft: usize,
        occupied_half: usize,
        num_taps: usize,
        stopband_db: f32,
    ) -> Self {
        let occupied_norm = occupied_half as f32 / n_fft.max(1) as f32;
        let half_transition = 0.5 * kaiser_transition_norm(num_taps, stopband_db);
        // Earliest cutoff that keeps the pass band clear of the carriers, and
        // the latest that keeps the stop band inside Nyquist.
        let earliest = occupied_norm + half_transition;
        let latest = 0.5 - half_transition;
        let cutoff = if earliest <= latest {
            earliest
        } else {
            0.5 * (occupied_norm + 0.5)
        };
        Self::new(cutoff, num_taps, stopband_db)
    }

    /// The shortest odd tap count whose Kaiser transition fits the null band of
    /// the same layout [`for_null_band`](Self::for_null_band) describes. Use it
    /// to size a filter, then check the result against the guard budget — if it
    /// does not fit, a longer guard (or a lower `stopband_db`) is the lever.
    pub fn taps_for_null_band(n_fft: usize, occupied_half: usize, stopband_db: f32) -> usize {
        let occupied_norm = occupied_half as f32 / n_fft.max(1) as f32;
        kaiser_num_taps(0.5 - occupied_norm, stopband_db)
    }

    /// Group delay in samples, `(num_taps − 1)/2` after the odd/≥3 clamp — the
    /// filter's reach on each side, and the quantity the guard must cover.
    pub fn group_delay(&self) -> usize {
        (self.num_taps.max(3) | 1) / 2
    }

    /// Approximate transition width as a fraction of `fs` (see
    /// [`kaiser_transition_norm`]).
    pub fn transition_norm(&self) -> f32 {
        kaiser_transition_norm(self.num_taps, self.stopband_db)
    }

    /// Whether the transition band stays clear of the occupied carriers *and*
    /// of Nyquist for the layout described by
    /// [`for_null_band`](Self::for_null_band) — i.e. the filter is long enough
    /// to reach its stop band inside the null band.
    pub fn transition_fits(&self, n_fft: usize, occupied_half: usize) -> bool {
        let occupied_norm = occupied_half as f32 / n_fft.max(1) as f32;
        self.transition_norm() <= 0.5 - occupied_norm
    }

    /// The frequency (fraction of `fs`) at which the stop band begins:
    /// `cutoff_norm + transition/2`. Beyond it the filter delivers its full
    /// `stopband_db` — this is where a mask's attenuation should be measured or
    /// specified, since the transition itself is deliberately unattenuated.
    pub fn stopband_edge_norm(&self) -> f32 {
        self.cutoff_norm + 0.5 * self.transition_norm()
    }

    /// Whether this filter's group delay and a `roll_off`-sample symbol taper
    /// both fit inside the guard samples a receiver at `backoff` discards:
    /// `roll_off + group_delay ≤ min(cp_len − backoff, backoff)`. Pass
    /// `roll_off = 0` when symbol windowing is off.
    pub fn fits_guard(&self, cp_len: usize, roll_off: usize, backoff: usize) -> bool {
        let slack = (cp_len.saturating_sub(backoff)).min(backoff);
        roll_off + self.group_delay() <= slack
    }

    /// Builds the filter instance this spec describes.
    pub fn filter(&self) -> FirLowpassIq {
        FirLowpassIq::design(self.num_taps, self.cutoff_norm, self.stopband_db)
    }

    /// Applies the filter in place across a whole assembled stream — same
    /// length, time-aligned, spanning symbol boundaries (that span is the point:
    /// it is what makes this a spectral filter rather than a per-symbol taper).
    pub fn apply(&self, stream: &mut [C32]) {
        self.filter().filter_aligned(stream);
    }
}
