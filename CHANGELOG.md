# Changelog

## v0.2.0 — corrections and didactic improvements

This release revises the original didactic implementation. The IBM Model 1
algorithm, module layout, and teaching goals are unchanged; the focus is on
correctness, robustness, and the extensions suggested by the project
specification (`spec/`).

### Fixed

- **Silent argument parsing.** Previously `--iterations abc` was parsed with
  `unwrap_or(default)` and the bad value was discarded without notice. CLI
  parsing is now strict: unknown flags, missing values, and unparsable numbers
  produce a clear `error:` message, print usage, and exit with a non-zero
  status. Covered by unit tests in `main.rs`.

- **Misleading "log-likelihood proxy".** The original metric was an unlabeled
  proxy whose averaging did not correspond to a real likelihood. It is now the
  length-normalized IBM Model 1 per-token log-likelihood (the `1 / (l + 1)`
  source-length factor is folded into each term, `epsilon` omitted as a
  constant), renamed `avg_log_likelihood_per_token`. A test asserts the EM
  guarantee that it never decreases between iterations.

- **ASCII-only tokenization.** Punctuation trimming used
  `char::is_ascii_punctuation`, leaving non-ASCII punctuation attached to
  tokens. Tokenization now trims any non-alphanumeric edge characters via
  `char::is_alphanumeric`, while preserving in-word accents (e.g. `petite`).

### Added

- **Test suite (spec §13).** Unit tests for vocabulary ID assignment, the
  `sum_f t(f|e) = 1` normalization invariant, probability movement during
  training, monotonic log-likelihood, deterministic JSON export, and
  tokenization. An integration test (`tests/integration.rs`) reproduces the
  spec's end-to-end expectation that `maison` aligns with `house` and `livre`
  with `book`.

- **JSON export of the learned table** (`--save-json <PATH>`), keyed by words,
  implemented by hand to keep the project dependency-free (std-only).
  `IBMModel1::to_json` (numeric IDs) and `to_named_json` (words) are available.

- **Pharaoh-format alignment output** (`--pharaoh`) and an optional
  `--no-null-align` mode that prefers real source words over `<NULL>` while
  still falling back to `<NULL>` when no real candidate has positive
  probability.

- **Load diagnostics.** `load_tsv` now returns `LoadStats` (pairs loaded,
  lines skipped) and emits warnings for malformed or empty lines.

- **`Vocabulary` helpers**: `id`, `contains`, and `is_empty` (the last also
  resolves the common Clippy `len`-without-`is_empty` lint).

- **`IBMModel1::row_sum`** for inspecting/testing per-source normalization.

### Changed

- Doc comments added across all modules, cross-referencing the relevant spec
  sections.
- `sorted_probabilities` now breaks ties deterministically (by `e`, then `f`)
  so output and tests are reproducible.
- `Cargo.toml` gains a real author field, repository/keywords metadata, and a
  release profile with LTO.

### Notes on EM behavior

With a uniform co-occurrence initialization, a single EM step can reproduce
the initial table on highly symmetric corpora (it is briefly a fixed point).
This is expected; the probabilities move once content-word asymmetries
compound over several iterations. The tests assert the realistic
multi-iteration behavior rather than a change after exactly one step.
