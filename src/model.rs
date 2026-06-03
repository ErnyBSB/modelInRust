use crate::corpus::SentencePair;
use crate::vocab::TokenId;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub struct IBMModel1 {
    /// Translation probabilities: table[e][f] = t(f | e).
    pub table: HashMap<TokenId, HashMap<TokenId, f64>>,
}

impl IBMModel1 {
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
            let uniform_probability = 1.0 / f_set.len() as f64;
            let mut row = HashMap::new();

            for f in f_set {
                row.insert(f, uniform_probability);
            }

            table.insert(e, row);
        }

        Self { table }
    }

    pub fn probability(&self, e: TokenId, f: TokenId) -> f64 {
        self.table
            .get(&e)
            .and_then(|row| row.get(&f))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn set_probability(&mut self, e: TokenId, f: TokenId, probability: f64) {
        self.table.entry(e).or_default().insert(f, probability);
    }

    pub fn sorted_probabilities(&self) -> Vec<(TokenId, TokenId, f64)> {
        let mut values = Vec::new();

        for (&e, row) in &self.table {
            for (&f, &probability) in row {
                values.push((e, f, probability));
            }
        }

        values.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        values
    }
}
