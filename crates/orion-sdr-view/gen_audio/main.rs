/// Standalone generator for built-in audio assets.
///
/// Writes:
///   crates/orion-sdr-view/assets/audio/cq_morse.wav  — morse-like tone bursts
///
/// Run from the workspace root:
///   cargo run --manifest-path crates/orion-sdr-view/gen_audio/Cargo.toml
///
/// File: 8000 Hz, mono, 32-bit float PCM.
/// A ~2 s silence is appended as the PTT inter-loop gap.

use std::f32::consts::PI;
use std::path::Path;

const FS: f32 = 8_000.0;
const GAP_SECS: f32 = 2.0;

fn write_wav(path: &Path, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: FS as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for &s in samples {
        writer.write_sample(s).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

fn silence(secs: f32) -> Vec<f32> {
    vec![0.0f32; (secs * FS) as usize]
}

fn sine_burst(freq_hz: f32, dur_secs: f32, amp: f32, phase: &mut f32) -> Vec<f32> {
    let n = (dur_secs * FS) as usize;
    let dphi = 2.0 * PI * freq_hz / FS;
    let ramp = (0.005 * FS) as usize;
    (0..n)
        .map(|i| {
            let env = if i < ramp {
                i as f32 / ramp as f32
            } else if i >= n - ramp {
                (n - i) as f32 / ramp as f32
            } else {
                1.0
            };
            let s = amp * env * phase.sin();
            *phase += dphi;
            if *phase > PI { *phase -= 2.0 * PI; }
            s
        })
        .collect()
}

const DIT: f32 = 0.080;
const DAH: f32 = 3.0 * DIT;
const TONE_HZ: f32 = 800.0;
const TONE_AMP: f32 = 0.8;

fn morse(c: char) -> &'static [bool] {
    match c {
        'A' => &[false, true],
        'B' => &[true, false, false, false],
        'C' => &[true, false, true, false],
        'D' => &[true, false, false],
        'E' => &[false],
        'F' => &[false, false, true, false],
        'G' => &[true, true, false],
        'H' => &[false, false, false, false],
        'I' => &[false, false],
        'J' => &[false, true, true, true],
        'K' => &[true, false, true],
        'L' => &[false, true, false, false],
        'M' => &[true, true],
        'N' => &[true, false],
        'O' => &[true, true, true],
        'P' => &[false, true, true, false],
        'Q' => &[true, true, false, true],
        'R' => &[false, true, false],
        'S' => &[false, false, false],
        'T' => &[true],
        'U' => &[false, false, true],
        'V' => &[false, false, false, true],
        'W' => &[false, true, true],
        'X' => &[true, false, false, true],
        'Y' => &[true, false, true, true],
        'Z' => &[true, true, false, false],
        '1' => &[false, true, true, true, true],
        '2' => &[false, false, true, true, true],
        '3' => &[false, false, false, true, true],
        '4' => &[false, false, false, false, true],
        '5' => &[false, false, false, false, false],
        '6' => &[true, false, false, false, false],
        '7' => &[true, true, false, false, false],
        '8' => &[true, true, true, false, false],
        '9' => &[true, true, true, true, false],
        '0' => &[true, true, true, true, true],
        _ => &[],
    }
}

fn gen_morse() -> Vec<f32> {
    let words: &[&str] = &["CQ", "CQ", "CQ", "DE", "N0GNR", "N0GNR", "K"];
    let mut out: Vec<f32> = Vec::new();
    let mut phase = 0.0f32;

    for (wi, word) in words.iter().enumerate() {
        for (ci, ch) in word.chars().enumerate() {
            let elements = morse(ch);
            for (ei, &is_dah) in elements.iter().enumerate() {
                let dur = if is_dah { DAH } else { DIT };
                out.extend(sine_burst(TONE_HZ, dur, TONE_AMP, &mut phase));
                if ei + 1 < elements.len() {
                    out.extend(silence(DIT));
                }
            }
            if ci + 1 < word.len() {
                out.extend(silence(2.0 * DIT));
            }
        }
        if wi + 1 < words.len() {
            out.extend(silence(4.0 * DIT));
        }
    }
    out.extend(silence(GAP_SECS));
    out
}

fn main() {
    let base = Path::new("crates/orion-sdr-view/assets/audio");

    let morse_samples = gen_morse();
    let morse_path = base.join("cq_morse.wav");
    write_wav(&morse_path, &morse_samples);
    println!(
        "Wrote {} ({:.1} s, {} samples)",
        morse_path.display(),
        morse_samples.len() as f32 / FS,
        morse_samples.len()
    );
}
