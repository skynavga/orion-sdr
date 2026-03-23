---
name: release
description: Push the prepared orion-sdr release tag and publish to crates.io. Run release-prep first.
allowed-tools: Bash
argument-hint: <version>  (e.g. 0.0.17)
---

Publish the orion-sdr release for version $ARGUMENTS.

VERSION = $ARGUMENTS  (without the leading "v")
TAG = v$ARGUMENTS

This skill assumes `/release-prep VERSION` has already been run successfully:
the version bump commit exists locally and the signed tag TAG exists locally.

## Step 1 — Verify preconditions

- Confirm the local tag TAG exists: `git tag -l TAG`
- Confirm the tag signature is valid: `git tag -v TAG`
- Confirm the commit the tag points to is ahead of origin/main:
  `git log origin/main..TAG --oneline`

If any check fails, stop and tell the user what is missing.

## Step 2 — Push commit and tag

Push in this order (commit first so the tag's target exists on the remote):

```
git push
git push origin TAG
```

This push triggers the GitHub Actions `publish.yml` workflow, which builds
wheels for all platforms and publishes to PyPI automatically.

## Step 3 — Publish to crates.io

```
cargo publish --allow-dirty
```

(`--allow-dirty` is needed because `.venv/`, `__pycache__/`, and `.pytest_cache/`
are untracked but excluded from the crate package by `Cargo.toml`'s `include`
list, so they do not affect what gets published.)

If publish fails with "already uploaded", the version is already on crates.io —
treat this as success and continue.

## Step 4 — Report

Tell the user:
- Commit and tag TAG have been pushed to GitHub
- The GitHub Actions workflow is now building wheels for all platforms and
  will publish them to PyPI automatically
- crates.io publish result (success or already-uploaded)
- Link to the Actions run: https://github.com/skynavga/orion-sdr/actions
- Link to the crates.io release: https://crates.io/crates/orion-sdr/VERSION
- Link to the PyPI release: https://pypi.org/project/orion-sdr/VERSION/
