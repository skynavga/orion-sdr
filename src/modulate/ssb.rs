// src/modulate/ssb.rs
use crate::dsp::{LpCascade, Rotator};
use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

#[allow(dead_code)]
pub struct SsbPhasingMod {
    fs: f32,
    audio_if_hz: f32,
    usb: bool,
    // Replaces two FirLowpass instances:
    lp_i: LpCascade,
    lp_q: LpCascade,
    // Rotators
    aud_nco: Rotator,
    rf_nco:  Rotator, // rf_hz==0.0 -> baseband
}

impl SsbPhasingMod {
    pub fn new(fs: f32, audio_bw_hz: f32, audio_if_hz: f32, rf_hz: f32, usb: bool) -> Self {
        // Slightly undercut fc to ensure good sideband suppression at edges
        let fc = audio_bw_hz * 0.9;
        Self {
            fs,
            audio_if_hz,
            usb,
            lp_i: LpCascade::design(fs, fc),
            lp_q: LpCascade::design(fs, fc),
            aud_nco: Rotator::new(audio_if_hz, fs),
            rf_nco:  Rotator::new(rf_hz, fs),
        }
    }
}

impl Block for SsbPhasingMod {
    type In = f32;   // audio
    type Out = C32;  // IQ

    #[inline]
    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());

        let side = if self.usb { 1.0 } else { -1.0 };
        let mut i = 0;
        let nn = n & !3; // small unroll

        while i < nn {
            // 0
            let p0 = self.aud_nco.next(); // (cos, sin)
            let i0 = self.lp_i.process(input[i]     * p0.re);
            let q0 = self.lp_q.process(input[i]     * p0.im);
            let z0 = C32::new(i0, side * q0);
            let r0 = self.rf_nco.next();
            output[i] = C32::new(
                z0.re.mul_add(r0.re, -z0.im * r0.im),
                z0.im.mul_add(r0.re,  z0.re * r0.im),
            );

            // 1
            let p1 = self.aud_nco.next();
            let i1 = self.lp_i.process(input[i+1] * p1.re);
            let q1 = self.lp_q.process(input[i+1] * p1.im);
            let z1 = C32::new(i1, side * q1);
            let r1 = self.rf_nco.next();
            output[i+1] = C32::new(
                z1.re.mul_add(r1.re, -z1.im * r1.im),
                z1.im.mul_add(r1.re,  z1.re * r1.im),
            );

            // 2
            let p2 = self.aud_nco.next();
            let i2 = self.lp_i.process(input[i+2] * p2.re);
            let q2 = self.lp_q.process(input[i+2] * p2.im);
            let z2 = C32::new(i2, side * q2);
            let r2 = self.rf_nco.next();
            output[i+2] = C32::new(
                z2.re.mul_add(r2.re, -z2.im * r2.im),
                z2.im.mul_add(r2.re,  z2.re * r2.im),
            );

            // 3
            let p3 = self.aud_nco.next();
            let i3 = self.lp_i.process(input[i+3] * p3.re);
            let q3 = self.lp_q.process(input[i+3] * p3.im);
            let z3 = C32::new(i3, side * q3);
            let r3 = self.rf_nco.next();
            output[i+3] = C32::new(
                z3.re.mul_add(r3.re, -z3.im * r3.im),
                z3.im.mul_add(r3.re,  z3.re * r3.im),
            );

            i += 4;
        }
        while i < n {
            let p = self.aud_nco.next();
            let ii = self.lp_i.process(input[i] * p.re);
            let qq = self.lp_q.process(input[i] * p.im);
            let z  = C32::new(ii, side * qq);
            let r  = self.rf_nco.next();
            output[i] = C32::new(
                z.re.mul_add(r.re, -z.im * r.im),
                z.im.mul_add(r.re,  z.re * r.im),
            );
            i += 1;
        }

        WorkReport { in_read: n, out_written: n }
    }
}
