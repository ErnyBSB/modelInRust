# IBM Model 1 in Rust: Didactic Implementation (v0.2)

This project is a small, educational Rust implementation of **IBM Model 1**,
one of the foundational probabilistic models in statistical machine
translation. It learns word-to-word translation probabilities `t(f | e)` from
a small parallel corpus using the **Expectation-Maximization (EM)** algorithm.

This is **version 0.2**, a corrected and extended revision of the original
didactic code. The pipeline and teaching goals are unchanged; the differences
are documented in `CHANGELOG.md` and summarized at the end of this file.

```text
parallel corpus -> vocabularies -> translation table t(f | e) -> EM training -> word alignments
```

Where `e` is a source word (English), `f` is a target word (French),
`t(f | e)` is the probability that `e` generates `f`, and `<NULL>` is the
empty source token (the empty cept `e0`).

## 1. Project structure

```text
ibm_model_1_rust/
├── Cargo.toml
├── README.md
├── CHANGELOG.md
├── data/
│   └── toy.tsv
├── src/
│   ├── main.rs       # CLI + orchestration
│   ├── corpus.rs     # reading + tokenization
│   ├── vocab.rs      # word <-> id mapping
│   ├── model.rs      # the t(f|e) table + JSON export
│   ├── trainer.rs    # EM (E-step / M-step) + log-likelihood
│   └── aligner.rs    # Viterbi-style alignment + Pharaoh format
└── tests/
    └── integration.rs
```

## 2. Running

```bash
cargo run -- --corpus data/toy.tsv --iterations 15 --top 8
```

Useful flags:

```text
--corpus <PATH>       TSV file: source<TAB>target  (default: data/toy.tsv)
--iterations <N>      Number of EM iterations        (default: 10)
--top <N>             Learned probabilities to print (default: 20)
--examples <N>        Example sentence alignments     (default: 3)
--save-json <PATH>    Write learned t(f|e) table as JSON (keyed by words)
--pharaoh             Also print alignments in Pharaoh format
--no-null-align       Prefer real source words over <NULL> in alignment
-h, --help            Show help
```

Run the tests:

```bash
cargo test
```

## 3. Example output

```text
Training report (log-likelihood per token should not decrease)
  iteration  1: avg log-likelihood/token = -1.726601
  ...
  iteration 15: avg log-likelihood/token = -1.509016

Top learned translation probabilities t(f | e)
   1. t(maison     | house     ) = 0.830938
   2. t(livre      | book      ) = 0.830938
   3. t(la         | the       ) = 0.499840
   4. t(le         | the       ) = 0.499840

Example alignments
Sentence pair 1
  la         -> the        score=0.499840
  maison     -> house      score=0.830938
  pharaoh: 0-0 1-1
```

## 4. The core EM idea

For each target word `f_j`, the responsibility of each source word `e_i` is

```text
delta(e_i, f_j) = t(f_j | e_i) / sum_i t(f_j | e_i)
```

These fractional counts are accumulated across the corpus (E-step), then
normalized to update the table (M-step):

```text
t(f | e) = count(e, f) / total(e)
```

The reported `avg log-likelihood/token` is the length-normalized IBM Model 1
log-likelihood per target token. EM guarantees it never decreases, so it
doubles as a convergence check during class.

## 5. Input format

UTF-8 TSV, two tab-separated columns. Lines beginning with `#` and blank lines
are ignored:

```text
the house	la maison
the book	le livre
```

## 6. What changed in v0.2 (summary)

Corrections and improvements over the original code, all driven by the project
specification (`spec/`):

1. **Unit and integration tests added** (spec §13): vocabulary IDs, `<NULL>`
   presence, the normalization invariant `sum_f t(f|e) = 1`, that EM moves the
   probabilities, that the log-likelihood is monotonically non-decreasing, and
   the end-to-end "maison aligns with house" expectation.
2. **Strict CLI parsing**: invalid numbers, missing values, and unknown flags
   now produce a clear error and a non-zero exit code instead of being
   silently ignored.
3. **Honest, length-normalized log-likelihood**: the metric is now a real
   per-token IBM Model 1 log-likelihood term (the `1/(l+1)` factor folded in),
   clearly named, instead of an unlabeled "proxy".
4. **Unicode-aware tokenization**: edge punctuation is stripped using
   `char::is_alphanumeric`, so French punctuation is handled, not just ASCII.
5. **JSON export** of the learned table (`--save-json`), keyed by words, with
   no external dependencies (still std-only), satisfying a suggested extension.
6. **Pharaoh-format alignment output** (`--pharaoh`) and an optional
   `--no-null-align` mode.
7. **Load statistics and warnings** for skipped/malformed lines.
8. **Documentation and metadata**: doc comments throughout and a real author
   field in `Cargo.toml`.

## 7. Intentionally out of scope

As in the original, and per spec §15: IBM Models 2–5, a full decoder, advanced
smoothing, log-space training, parallelism, embeddings, and neural models are
deliberately omitted to keep the EM process inspectable.

## 8. License

MIT.
