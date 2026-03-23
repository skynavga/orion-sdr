---
name: release-prep
description: Bump orion-sdr version, update CHANGELOG, run tests, commit, and create a signed tag — but do not push or publish.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
argument-hint: <new-version>  (e.g. 0.0.17)
---

Prepare an orion-sdr release for version $ARGUMENTS.

The previous version is the one currently in `Cargo.toml`. Determine it by
reading that file. Call it OLD_VERSION. The new version is $ARGUMENTS;
call it NEW_VERSION.

## Step 1 — Verify preconditions

- Confirm current branch is not `main`. If it is, stop and tell the user.
- Confirm the working tree is clean (`git status`). If there are uncommitted
  changes, stop and tell the user.
- Confirm NEW_VERSION > OLD_VERSION (simple string check is fine).

## Step 2 — Bump version strings

Update OLD_VERSION → NEW_VERSION in every file listed below. Read each file
before editing it.

| File | What to change |
|------|----------------|
| `Cargo.toml` | `version = "OLD_VERSION"` |
| `pyproject.toml` | `version = "OLD_VERSION"` |
| `README.md` | `Pre-alpha (vOLD_VERSION)` and `as of vOLD_VERSION` in the Features doc link |
| `docs/features.md` | `# Features (as of vOLD_VERSION)` heading |
| `docs/performance.md` | `## vOLD_VERSION Results` heading |
| `CLAUDE.md` | `vOLD_VERSION), edition 2024` in the project description |
| `memory/MEMORY.md` at `~/.claude/projects/.../memory/MEMORY.md` | version in the Project Summary line |

To find the memory file path, use Glob to search for `MEMORY.md` under
`~/.claude/projects/` and pick the one for this project.

## Step 3 — Prepend CHANGELOG entry

Read `CHANGELOG.md`. Insert a new `## [NEW_VERSION] - TODAY` section
immediately before the existing `## [OLD_VERSION]` section.

TODAY is the current date in YYYY-MM-DD format (use `date +%F` via Bash).

The entry should document what actually changed since OLD_VERSION. Inspect
`git log OLD_VERSION_TAG..HEAD --oneline` (where OLD_VERSION_TAG is
`vOLD_VERSION`) to find the commits, then write a concise Added/Changed/Fixed
list. If there are no real changes (test release), write a minimal entry such as:

```
## [NEW_VERSION] - TODAY

### Changed

- (describe changes here based on git log)
```

## Step 4 — Run tests

Run both test suites and verify they all pass:

```
cargo test --release
.venv/bin/pytest -q
```

If either suite fails, stop and report the failure. Do not proceed.

## Step 5 — Commit

Stage only the files changed in steps 2 and 3 (never `git add -A`):

```
git add Cargo.toml Cargo.lock pyproject.toml README.md \
        docs/features.md docs/performance.md CLAUDE.md CHANGELOG.md
```

Commit with message: `Bump version to NEW_VERSION`

Do not include a co-author trailer.

## Step 6 — Merge to main via PR

Push the current branch to origin if it has no upstream yet:

```
git push -u origin HEAD
```

Check whether a PR already exists for the current branch:

```
gh pr list --head CURRENT_BRANCH --state open
```

If no open PR exists, create one. Inspect `git log main..HEAD --oneline` to
understand all changes in the branch, then write a concise BLUF-style summary
(one short paragraph) covering all significant changes. Follow it with a
"Release prep for NEW_VERSION." line. Example format:

```
<One short paragraph summarizing all significant changes in the branch.>

Release prep for NEW_VERSION.
```

Merge the PR:

```
gh pr merge --merge --delete-branch
```

Switch to `main` and pull so the local branch is up to date:

```
git checkout main
git pull
```

Confirm the current branch is now `main` before proceeding.

## Step 7 — Create signed tag

```
git tag -s vNEW_VERSION -m "Release NEW_VERSION"
```

Then verify it:
```
git tag -v vNEW_VERSION
```

Confirm the GPG signature is good before reporting success.

## Step 8 — Report

Tell the user:
- What version was bumped (OLD → NEW)
- That all tests passed
- That the commit and signed tag are ready locally
- That the next step is `/release NEW_VERSION` to push and publish to crates.io
