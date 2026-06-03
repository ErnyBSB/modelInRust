//! The IBM Model 1 translation table `t(f | e)` and operations on it.
//!
//! The table is stored sparsely as `HashMap<TokenId, HashMap<TokenId, f64>>`,
//! read as `table[e][f] = t(f | e)` (spec, section 5, alternative B). This is
//! memory-friendly and maps directly onto the conditional-probability notation.

use crate::corpus::SentencePair;
use crate::vocab::TokenId;
use std::collections::{HashMap, HashSet};

/// Sparse translation table holding the learned parameters of IBM Model 1.
#[derive(Debug, Default)]
pub struct IBMModel1 {
    /// `table[e][f] = t(f | e)`.
    pub table: HashMap<TokenId, HashMap<TokenId, f64>>,
}

#[allow(dead_code)] // public API; some methods exercised only by tests/consumers
impl IBMModel1 {
    /// Initialize `t(f | e)` uniformly from sentence-level co-occurrences.
    ///
    /// For each source word `e`, the probability mass is spread evenly across
    /// only the target words that actually co-occur with it, rather than the
    /// whole target vocabulary (spec, section 6).
    pub fn new_uniform_from_corpus(corpus: &[SentencePair]) -> Self {
        let mut cooccurrences: HashMap<TokenId, HashSet<TokenId>> = HashMap::new();

        for pair in corpus {
            for &e in &pair.source {
                let entry = cooccurrences.entry(e).or_default();
                for &f in &pair.target {
                    entry.insert(f);
                }
            }
        }

        let mut table = HashMap::new();
        for (e, f_set) in cooccurrences {
            let uniform = 1.0 / f_set.len() as f64;
            let row: HashMap<TokenId, f64> = f_set.into_iter().map(|f| (f, uniform)).collect();
            table.insert(e, row);
        }

        Self { table }
    }

    /// `t(f | e)`, or `0.0` if the pair was never observed.
    pub fn probability(&self, e: TokenId, f: TokenId) -> f64 {
        self.table
            .get(&e)
            .and_then(|row| row.get(&f))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set `t(f | e)`.
    pub fn set_probability(&mut self, e: TokenId, f: TokenId, probability: f64) {
        self.table.entry(e).or_default().insert(f, probability);
    }

    /// All `(e, f, t(f | e))` triples sorted by descending probability.
    pub fn sorted_probabilities(&self) -> Vec<(TokenId, TokenId, f64)> {
        let mut values: Vec<(TokenId, TokenId, f64)> = self
            .table
            .iter()
            .flat_map(|(&e, row)| row.iter().map(move |(&f, &p)| (e, f, p)))
            .collect();

        values.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        values
    }

    /// Sum of `t(f | e)` over all `f` for a given source word `e`.
    ///
    /// After any complete M-step this must equal `1.0` for every `e` that has
    /// a row, which is exactly the invariant exercised by the unit tests
    /// (spec, section 13, item 3).
    pub fn row_sum(&self, e: TokenId) -> f64 {
        self.table
            .get(&e)
            .map(|row| row.values().sum())
            .unwrap_or(0.0)
    }

    /// Serialize the table to a minimal, dependency-free JSON string of the
    /// shape `{ "e_id": { "f_id": prob, ... }, ... }`.
    ///
    /// Implementing this by hand keeps the project free of `serde` while still
    /// satisfying the "save the trained table" extension (spec, section 13 /
    /// suggested extensions). Use [`crate::model::IBMModel1::to_named_json`]
    /// for a human-readable variant.
    pub fn to_json(&self) -> String {
        let mut rows: Vec<(TokenId, &HashMap<TokenId, f64>)> =
            self.table.iter().map(|(&e, row)| (e, row)).collect();
        rows.sort_by_key(|(e, _)| *e);

        let mut out = String::from("{");
        for (i, (e, row)) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{e}\":{{"));

            let mut cells: Vec<(TokenId, f64)> = row.iter().map(|(&f, &p)| (f, p)).collect();
            cells.sort_by_key(|(f, _)| *f);
            for (j, (f, p)) in cells.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!("\"{f}\":{p:.6}"));
            }
            out.push('}');
        }
        out.push('}');
        out
    }

    /// Human-readable JSON keyed by the actual words instead of numeric IDs:
    /// `{ "house": { "maison": 0.82, ... }, ... }`. Requires the two
    /// vocabularies used during training.
    pub fn to_named_json(
        &self,
        source_vocab: &crate::vocab::Vocabulary,
        target_vocab: &crate::vocab::Vocabulary,
    ) -> String {
        let mut rows: Vec<(TokenId, &HashMap<TokenId, f64>)> =
            self.table.iter().map(|(&e, row)| (e, row)).collect();
        rows.sort_by_key(|(e, _)| *e);

        let mut out = String::from("{");
        for (i, (e, row)) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\":{{", json_escape(source_vocab.word(*e))));

            let mut cells: Vec<(TokenId, f64)> = row.iter().map(|(&f, &p)| (f, p)).collect();
            cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, (f, p)) in cells.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!("\"{}\":{p:.6}", json_escape(target_vocab.word(*f))));
            }
            out.push('}');
        }
        out.push('}');
        out
    }
}

/// Minimal JSON string escaping for the characters that can appear in tokens.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::SentencePair;

    fn toy_corpus() -> Vec<SentencePair> {
        // source already includes <NULL> = id 0
        vec![
            SentencePair {
                source: vec![0, 1, 2],
                target: vec![10, 11],
            },
            SentencePair {
                source: vec![0, 1, 3],
                target: vec![10, 12],
            },
        ]
    }

    #[test]
    fn uniform_init_rows_sum_to_one() {
        let model = IBMModel1::new_uniform_from_corpus(&toy_corpus());
        for &e in &[0usize, 1, 2, 3] {
            assert!((model.row_sum(e) - 1.0).abs() < 1e-9, "row {e} not normalized");
        }
    }

    #[test]
    fn json_export_is_deterministic_and_sorted() {
        let mut model = IBMModel1::default();
        model.set_probability(1, 20, 0.25);
        model.set_probability(1, 10, 0.75);
        let json = model.to_json();
        // f=10 must appear before f=20 because keys are sorted.
        let pos10 = json.find("\"10\"").unwrap();
        let pos20 = json.find("\"20\"").unwrap();
        assert!(pos10 < pos20);
    }
}
