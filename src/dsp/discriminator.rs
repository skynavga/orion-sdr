// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::LpCascade;
use crate::util::atan2_approx;
use num_complex::Complex32 as C32;

/// Quadrature (discriminator) core shared by FM and PM demod: angle of the
/// product of the current sample and the conjugate of the previous one,
/// scaled by `k`, run through the caller's post-filter.
#[inline(always)]
pub(crate) fn quadrature_discriminate(z: C32, prev: C32, k: f32, post_lp: &mut LpCascade) -> f32 {
    let prod = C32::new(
        z.re * prev.re + z.im * prev.im,
        z.im * prev.re - z.re * prev.im,
    );
    post_lp.process(atan2_approx(prod.im, prod.re) * k)
}
