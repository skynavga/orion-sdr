use num_complex::Complex32 as C32;

#[derive(Debug, Clone, Copy, Default)]
pub struct WorkReport {
    pub in_read: usize,
    pub out_written: usize,
}

pub trait Block {
    type In;
    type Out;
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport;

    /// Optional specialized path; default forwards to `process`.
    #[inline]
    fn process_into(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        self.process(input, output)
    }
}

/// Audio (f32) -> IQ (C32)
pub struct AudioToIqChain<B: Block<In = f32, Out = C32>> {
    block: B,
    out: Vec<C32>,
}
impl<B: Block<In = f32, Out = C32>> AudioToIqChain<B> {
    pub fn new(block: B) -> Self {
        Self { block, out: Vec::new() }
    }
    /// Original convenience API (takes ownership of Vec)
    pub fn process(&mut self, input: Vec<f32>) -> Vec<C32> {
        self.process_ref(&input)
    }
    /// Borrowed input to avoid clone in hot paths.
    pub fn process_ref(&mut self, input: &[f32]) -> Vec<C32> {
        if self.out.len() < input.len() {
            self.out.resize(input.len(), C32::new(0.0, 0.0));
        }
        let n = input.len();
        let _wr = self.block.process_into(input, &mut self.out[..n]);
        self.out[..n].to_vec()
    }
    /// Fully preallocated path.
    pub fn process_into(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        self.block.process_into(input, output)
    }
}

/// IQ (C32) -> Audio (f32)
pub struct IqToAudioChain<B: Block<In = C32, Out = f32>> {
    block: B,
    out: Vec<f32>,
}
impl<B: Block<In = C32, Out = f32>> IqToAudioChain<B> {
    pub fn new(block: B) -> Self {
        Self { block, out: Vec::new() }
    }
    pub fn process(&mut self, input: Vec<C32>) -> Vec<f32> {
        self.process_ref(&input)
    }
    pub fn process_ref(&mut self, input: &[C32]) -> Vec<f32> {
        if self.out.len() < input.len() {
            self.out.resize(input.len(), 0.0);
        }
        let n = input.len();
        let _wr = self.block.process_into(input, &mut self.out[..n]);
        self.out[..n].to_vec()
    }
    pub fn process_into(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        self.block.process_into(input, output)
    }
}
