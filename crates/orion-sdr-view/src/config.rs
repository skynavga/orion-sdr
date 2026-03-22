use serde::Deserialize;

// ── Serde structs ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DisplayConfig {
    pub db_min: Option<f32>,
    pub db_max: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TestToneConfig {
    pub freq_hz:    Option<f32>,
    pub noise_amp:  Option<f32>,
    pub amp_max:    Option<f32>,
    pub ramp_secs:  Option<f32>,
    pub pause_secs: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct AmDsbConfig {
    pub carrier_hz:    Option<f32>,
    pub mod_index:     Option<f32>,
    pub loop_gap_secs: Option<f32>,
    pub noise_amp:     Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct SourcesConfig {
    pub test_tone: Option<TestToneConfig>,
    pub am_dsb:    Option<AmDsbConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ViewConfig {
    pub display: Option<DisplayConfig>,
    pub sources: Option<SourcesConfig>,
}

// Private top-level wrapper matching `view:` key
#[derive(Debug, Deserialize)]
struct ConfigFile {
    pub view: Option<ViewConfig>,
}

// ── Built-in defaults ─────────────────────────────────────────────────────────

pub struct Defaults;
impl Defaults {
    pub const DB_MIN:        f32 = -80.0;
    pub const DB_MAX:        f32 = -20.0;
    pub const FREQ_HZ:       f32 = 3_000.0;
    pub const NOISE_AMP:     f32 = 0.05;
    pub const AMP_MAX:       f32 = 0.65;
    pub const RAMP_SECS:     f32 = 3.0;
    pub const PAUSE_SECS:    f32 = 7.0;
    pub const CARRIER_HZ:    f32 = 10_000.0;
    pub const MOD_INDEX:     f32 = 1.0;
    pub const LOOP_GAP_SECS: f32 = 7.0;
    pub const AM_NOISE_AMP:  f32 = 0.05;
}

// ── ViewConfig: three-tier loader + accessors ─────────────────────────────────

impl ViewConfig {
    /// Three-tier resolver:
    /// 1. `--config <path>` (hard-fail on error)
    /// 2. `.orionsdr.yaml` in CWD (soft-warn on error, skip if absent)
    /// 3. Built-in defaults (returns empty ViewConfig)
    pub fn load(explicit_path: Option<std::path::PathBuf>) -> Self {
        if let Some(p) = explicit_path {
            return Self::from_path(&p, true);
        }
        let cwd = std::path::PathBuf::from(".orionsdr.yaml");
        if cwd.exists() {
            return Self::from_path(&cwd, false);
        }
        Self::empty()
    }

    fn from_path(path: &std::path::Path, hard_fail: bool) -> Self {
        let content = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("orion-sdr-view: error reading config {:?}: {}", path, e);
                if hard_fail {
                    std::process::exit(1);
                }
                return Self::empty();
            }
        };
        match serde_yaml::from_str::<ConfigFile>(&content) {
            Ok(cf) => cf.view.unwrap_or_else(Self::empty),
            Err(e) => {
                eprintln!("orion-sdr-view: error parsing config {:?}: {}", path, e);
                if hard_fail {
                    std::process::exit(1);
                }
                Self::empty()
            }
        }
    }

    fn empty() -> Self {
        ViewConfig { display: None, sources: None }
    }

    // ── Convenience accessors ─────────────────────────────────────────────

    pub fn db_min(&self) -> f32 {
        self.display.as_ref().and_then(|d| d.db_min).unwrap_or(Defaults::DB_MIN)
    }
    pub fn db_max(&self) -> f32 {
        self.display.as_ref().and_then(|d| d.db_max).unwrap_or(Defaults::DB_MAX)
    }
    pub fn freq_hz(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.test_tone.as_ref())
            .and_then(|t| t.freq_hz)
            .unwrap_or(Defaults::FREQ_HZ)
    }
    pub fn noise_amp(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.test_tone.as_ref())
            .and_then(|t| t.noise_amp)
            .unwrap_or(Defaults::NOISE_AMP)
    }
    pub fn amp_max(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.test_tone.as_ref())
            .and_then(|t| t.amp_max)
            .unwrap_or(Defaults::AMP_MAX)
    }
    pub fn ramp_secs(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.test_tone.as_ref())
            .and_then(|t| t.ramp_secs)
            .unwrap_or(Defaults::RAMP_SECS)
    }
    pub fn pause_secs(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.test_tone.as_ref())
            .and_then(|t| t.pause_secs)
            .unwrap_or(Defaults::PAUSE_SECS)
    }
    pub fn carrier_hz(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.am_dsb.as_ref())
            .and_then(|a| a.carrier_hz)
            .unwrap_or(Defaults::CARRIER_HZ)
    }
    pub fn mod_index(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.am_dsb.as_ref())
            .and_then(|a| a.mod_index)
            .unwrap_or(Defaults::MOD_INDEX)
    }
    pub fn loop_gap_secs(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.am_dsb.as_ref())
            .and_then(|a| a.loop_gap_secs)
            .unwrap_or(Defaults::LOOP_GAP_SECS)
    }
    pub fn am_noise_amp(&self) -> f32 {
        self.sources.as_ref()
            .and_then(|s| s.am_dsb.as_ref())
            .and_then(|a| a.noise_amp)
            .unwrap_or(Defaults::AM_NOISE_AMP)
    }
}
