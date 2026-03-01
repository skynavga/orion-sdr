use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{FirLowpass, Rotator};
 
 // FM Quadrature Demod
#[derive(Debug, Clone)]
 pub struct FmQuadratureDemod {
    fs: f32,
    k: f32,
    // optional translator
    xf: Option<Rotator>,
    prev: C32,
    // audio postfilter
    post_lp: FirLowpass,
 }

impl FmQuadratureDemod {

    pub fn new(fs: f32, dev_hz: f32, audio_bw_hz: f32) -> Self {
        let k = 1.0 / dev_hz.max(1.0);
        let post_lp = FirLowpass::design(fs, audio_bw_hz * 0.9, audio_bw_hz * 0.3);
         Self {
            fs,
            k,
            xf: None,
            prev: C32::new(1.0, 0.0),
            post_lp,
         }
     }

     pub fn with_translate(mut self, freq_hz: f32) -> Self {
        self.xf = Some(Rotator::new(freq_hz, self.fs)); self
    }

}
 
impl Block for FmQuadratureDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut ytmp = vec![0.0f32; n];
        if let Some(r) = &mut self.xf {
            for i in 0..n {
                let z = input[i] * r.next().conj();
                let prod = C32::new(
                    z.re * self.prev.re + z.im * self.prev.im,
                    z.im * self.prev.re - z.re * self.prev.im,
                );
                ytmp[i] = prod.im.atan2(prod.re) * self.k;
                self.prev = z;
            }
        } else {
            for i in 0..n {
                let z = input[i];
                let prod = C32::new(
                    z.re * self.prev.re + z.im * self.prev.im,
                    z.im * self.prev.re - z.re * self.prev.im,
                );
                ytmp[i] = prod.im.atan2(prod.re) * self.k;
                self.prev = z;
            }
        }

        // LP -> output
        self.post_lp.process(&ytmp[..], &mut output[..n]);

        WorkReport { in_read: n, out_written: n }
    }

}
