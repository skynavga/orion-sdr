---
name: release
description: Push the prepared orion-sdr release tag and publish to crates.io. Run release-prep first.
allowed-tools: Bash, Read, Write
argument-hint: <version>  (e.g. 0.0.17)
---

# Release

Publish the orion-sdr release for version $ARGUMENTS.

VERSION = $ARGUMENTS  (without the leading "v")
TAG = v$ARGUMENTS

This skill assumes `/release-prep VERSION` has already been run successfully:
the version bump commit exists locally and the signed tag TAG exists locally.

## Step 1 — Verify preconditions

- Confirm the local tag TAG exists: `git tag -l TAG`
- Confirm the tag signature is valid: `git tag -v TAG`
- Confirm the tag is **not behind** origin/main — i.e. it contains everything
  already published there: `git log TAG..origin/main --oneline` must print
  **nothing**. If it prints commits, the tag was cut before they landed; stop.

If any check fails, stop and tell the user what is missing.

**The tag being *equal* to origin/main is the normal case, not a problem.**
`/release-prep` merges the release branch through a GitHub PR, so by the time
this skill runs, origin/main already contains the tagged commit and the tag can
never be *ahead*. Check the direction above (`TAG..origin/main`), which is the
one that catches a stale tag; the reverse direction (`origin/main..TAG`) is
informational only — empty means the PR flow, non-empty means the tag was cut on
an unmerged branch, and both are fine.

To see the whole picture at once:

```bash
git rev-parse TAG^{commit} origin/main
```

Two identical SHAs is the expected result of the PR flow.

## Step 2 — Push commit and tag

Push in this order (commit first so the tag's target exists on the remote):

```bash
git push
git push origin TAG
```

After the PR flow the first push reports `Everything up-to-date` — the commit
went to origin with the merge. That is expected; the tag push is the one that
does the work here.

This push triggers the GitHub Actions `publish.yml` workflow, which builds
wheels for all platforms and publishes to PyPI automatically.

## Step 3 — Publish to crates.io

```bash
cargo publish --allow-dirty
```

(`--allow-dirty` is needed because `.venv/`, `__pycache__/`, and `.pytest_cache/`
are untracked but excluded from the crate package by `Cargo.toml`'s `include`
list, so they do not affect what gets published.)

If publish fails with "already uploaded", the version is already on crates.io —
treat this as success and continue.

## Step 4 — Cut the GitHub release

Not every tag gets a GitHub release, so the notes must cover **everything since
the last release that exists** — which may span several tags — not just the last
tag. Determine that boundary first:

```bash
gh release list --limit 1
gh release view --json tagName -q .tagName
```

Call the result PREV_TAG. If there is no prior release at all, use the repo's
first commit as the boundary.

### Gather the delta

- Commits: `git log PREV_TAG..TAG --oneline`
- Merged PRs in the range: the merge commits in that log name them
  (`Merge pull request #NN from ...`). Read each with `gh pr view NN` for the
  problem statement and any measured numbers.
- CHANGELOG: read the `## [x.y.z]` sections in `CHANGELOG.md` for **every**
  version in `(PREV_TAG, TAG]`, not only NEW_VERSION.
- Scale of the change: `git diff PREV_TAG..TAG --stat | tail -1`

### Match the house style

Read the previous two releases before drafting — they are the style reference:

```bash
gh release view PREV_TAG
```

Title: `vVERSION — <short phrase naming what changed>`, e.g.
`v0.0.59 — Streaming receiver: frame ordering and carrier tracking`. Describe
the change, not the version.

Body, in this order (omit a section only when it would be empty):

- **An opening paragraph**, BLUF. Name the PR(s) — "This release covers
  **PR #47**, …" — then state what changed and why it mattered. A second
  paragraph is warranted when the changes share a theme worth stating.
- **`## Highlights`** — bullets, each opening with a **bolded claim** and then
  the reasoning behind it. Prefer measured numbers over adjectives, and quote
  the same figures the CHANGELOG does. Include the notable *rejected*
  alternatives where they explain the design.
- **`## What it buys`** and/or **`## What it costs`** — the measured effect.
  Use a table for before/after curves. Say "Nothing" plainly when a change is
  free rather than omitting the section.
- **`## Breaking`** — API changes, waveform changes, behaviour a caller may have
  tuned against. Say "Nothing." explicitly when there are none.
- **`## Also`** — secondary fixes, test-harness changes, doc updates.
- The closing line, verbatim:

  ```text
  See [CHANGELOG.md](https://github.com/skynavga/orion-sdr/blob/main/CHANGELOG.md)
  for the per-version detail.
  ```

Wrap prose at 80 columns, as the existing release bodies do.

**Do not append a `Co-Authored-By:` trailer, a "Generated with Claude Code"
line, or any other attribution footer to the release body.** The closing
CHANGELOG line above is the last thing in it. This matches the same rule for
commit messages and PR descriptions.

### Create it

Write the body to a scratch file (not into the repo) and create the release
against the already-pushed tag:

```bash
gh release create TAG --title "vVERSION — <phrase>" --notes-file /tmp/orion-sdr-release-VERSION.md --latest
```

Then verify and clean up:

```bash
gh release view TAG
rm /tmp/orion-sdr-release-VERSION.md
```

If a release for TAG already exists, do not create a second one — show the user
the existing release and ask whether to edit it (`gh release edit TAG`).

## Step 5 — Report

Tell the user:

- Commit and tag TAG have been pushed to GitHub
- The GitHub Actions workflow is now building wheels for all platforms and
  will publish them to PyPI automatically
- crates.io publish result (success or already-uploaded)
- Which tag range the GitHub release notes cover (PREV_TAG..TAG)
- Link to the Actions run: <https://github.com/skynavga/orion-sdr/actions>
- Link to the GitHub release:
  <https://github.com/skynavga/orion-sdr/releases/tag/TAG>
- Link to the crates.io release: <https://crates.io/crates/orion-sdr/VERSION>
- Link to the PyPI release: <https://pypi.org/project/orion-sdr/VERSION/>
