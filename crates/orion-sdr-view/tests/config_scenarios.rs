//! Exercises the three-tier config loading logic without launching the GUI.

use std::io::Write;
use tempfile::NamedTempFile;

// Pull in the config module via the binary's source path trick.
// We compile the module directly here since there's no lib target.
include!("../src/config.rs");

fn defaults_all_match(cfg: &ViewConfig) {
    assert_eq!(cfg.db_min(),        Defaults::DB_MIN,        "db_min");
    assert_eq!(cfg.db_max(),        Defaults::DB_MAX,        "db_max");
    assert_eq!(cfg.freq_hz(),       Defaults::FREQ_HZ,       "freq_hz");
    assert_eq!(cfg.noise_amp(),     Defaults::NOISE_AMP,     "noise_amp");
    assert_eq!(cfg.amp_max(),       Defaults::AMP_MAX,       "amp_max");
    assert_eq!(cfg.ramp_secs(),     Defaults::RAMP_SECS,     "ramp_secs");
    assert_eq!(cfg.pause_secs(),    Defaults::PAUSE_SECS,    "pause_secs");
    assert_eq!(cfg.carrier_hz(),    Defaults::CARRIER_HZ,    "carrier_hz");
    assert_eq!(cfg.mod_index(),     Defaults::MOD_INDEX,     "mod_index");
    assert_eq!(cfg.loop_gap_secs(), Defaults::LOOP_GAP_SECS, "loop_gap_secs");
    assert_eq!(cfg.am_noise_amp(),  Defaults::AM_NOISE_AMP,  "am_noise_amp");
}

// ── Scenario 1: explicit --config with full YAML ─────────────────────────────

#[test]
fn explicit_config_full() {
    let yaml = r#"
view:
  display:
    db_min: -100.0
    db_max: -10.0
  sources:
    test_tone:
      freq_hz:    5000.0
      noise_amp:  0.10
      amp_max:    0.80
      ramp_secs:  2.0
      pause_secs: 5.0
    am_dsb:
      carrier_hz:    15000.0
      mod_index:     0.5
      loop_gap_secs: 3.0
      noise_amp:     0.02
"#;
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(yaml.as_bytes()).unwrap();

    let cfg = ViewConfig::load(Some(f.path().to_path_buf()));
    assert_eq!(cfg.db_min(),        -100.0);
    assert_eq!(cfg.db_max(),         -10.0);
    assert_eq!(cfg.freq_hz(),       5000.0);
    assert_eq!(cfg.noise_amp(),       0.10);
    assert_eq!(cfg.amp_max(),         0.80);
    assert_eq!(cfg.ramp_secs(),       2.0);
    assert_eq!(cfg.pause_secs(),      5.0);
    assert_eq!(cfg.carrier_hz(),  15000.0);
    assert_eq!(cfg.mod_index(),       0.5);
    assert_eq!(cfg.loop_gap_secs(),   3.0);
    assert_eq!(cfg.am_noise_amp(),   0.02);
}

// ── Scenario 3: explicit --config with partial YAML → overrides + defaults ────

#[test]
fn explicit_config_partial() {
    let yaml = "view:\n  display:\n    db_min: -120.0\n";
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(yaml.as_bytes()).unwrap();

    let cfg = ViewConfig::load(Some(f.path().to_path_buf()));
    assert_eq!(cfg.db_min(), -120.0);
    // everything else falls back to defaults
    assert_eq!(cfg.db_max(),        Defaults::DB_MAX);
    assert_eq!(cfg.freq_hz(),       Defaults::FREQ_HZ);
    assert_eq!(cfg.carrier_hz(),    Defaults::CARRIER_HZ);
}

// ── Scenario 4: explicit --config missing file → exit(1) ──────────────────────
// (Can't test process::exit in-process; verified manually via CLI)

// ── Scenario 5: explicit --config invalid YAML → exit(1) ─────────────────────
// (Same — tested manually)

// ── Scenario 6: CWD .orionsdr.yaml present and valid ─────────────────────────
// ── Scenario 7: CWD .orionsdr.yaml invalid YAML → soft-warn, use defaults ────
//
// CWD tests mutate the process working directory, so they must run serially.
// We combine them under one test guarded by a static mutex.

#[test]
fn cwd_config_scenarios() {
    use std::sync::Mutex;
    static CWD_LOCK: Mutex<()> = Mutex::new(());
    let _guard = CWD_LOCK.lock().unwrap();

    let orig = std::env::current_dir().unwrap();

    // 6a: valid .orionsdr.yaml
    {
        let dir = tempfile::tempdir().unwrap();
        let yaml = "view:\n  display:\n    db_max: -5.0\n";
        std::fs::write(dir.path().join(".orionsdr.yaml"), yaml).unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let cfg = ViewConfig::load(None);
        std::env::set_current_dir(&orig).unwrap();

        assert_eq!(cfg.db_max(), -5.0, "CWD config: db_max should be -5.0");
        assert_eq!(cfg.db_min(), Defaults::DB_MIN, "CWD config: db_min should be default");
    }

    // 6b: invalid .orionsdr.yaml → soft-warn, fall back to defaults
    {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".orionsdr.yaml"), b"{ this is not: [valid yaml").unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let cfg = ViewConfig::load(None);
        std::env::set_current_dir(&orig).unwrap();

        defaults_all_match(&cfg);
    }

    // 6c: no .orionsdr.yaml in CWD → all defaults
    {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let cfg = ViewConfig::load(None);
        std::env::set_current_dir(&orig).unwrap();

        defaults_all_match(&cfg);
    }
}

// ── Scenario 8: YAML with unknown top-level keys → silently ignored ───────────

#[test]
fn unknown_keys_ignored() {
    let yaml = r#"
view:
  display:
    db_min: -90.0
  future_key: ignored_value
library:
  some_setting: 42
"#;
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(yaml.as_bytes()).unwrap();

    let cfg = ViewConfig::load(Some(f.path().to_path_buf()));
    assert_eq!(cfg.db_min(), -90.0);
    assert_eq!(cfg.db_max(), Defaults::DB_MAX);
}

// ── Scenario 9: YAML with missing `view:` key → all defaults ─────────────────

#[test]
fn missing_view_key_uses_defaults() {
    let yaml = "# no view key here\nlibrary:\n  x: 1\n";
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(yaml.as_bytes()).unwrap();

    let cfg = ViewConfig::load(Some(f.path().to_path_buf()));
    defaults_all_match(&cfg);
}
