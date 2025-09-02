use crate::core::{Block, WorkReport};

/// 1st-order DC blocker (high-pass): y[n] = x[n] - x[n-1] + r*y[n-1]
#[derive(Debug, Clone)]
pub struct DcBlocker {
    r: f32,
    x1: f32,
    y1: f32,
}

impl DcBlocker {
    pub fn new(fs: f32, cut_hz: f32) -> Self {
        // Simple approximation for small cut-off
        let r = (1.0 - 2.0 * core::f32::consts::PI * (cut_hz.max(0.1) / fs)).clamp(0.0, 0.9999);
        Self { r, x1: 0.0, y1: 0.0 }
    }

    /// Fast per-sample DC-block step. High-performance inner-loop helper.
    #[inline(always)]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        // y = x - x1 + r*y1; then update state
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

impl Block for DcBlocker {
    type In = f32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut x1 = self.x1;
        let mut y1 = self.y1;
        let r = self.r;
        for i in 0..n {
            let x = input[i];
            let y = x - x1 + r * y1;
            output[i] = y;
            x1 = x;
            y1 = y;
        }
        self.x1 = x1; self.y1 = y1;
        WorkReport { in_read: n, out_written: n }
    }
}
