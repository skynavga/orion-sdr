/// orion-sdr: early scaffold.
/// `version()` is here so dependents can sanity-check linkage.
pub fn version() -> &'static str { env!("CARGO_PKG_VERSION") }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = version();
        assert_eq!(result, "0.0.1");
    }
}
