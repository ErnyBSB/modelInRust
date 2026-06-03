//! EM training for IBM Model 1 (spec, sections 7 and 8).
//!
//! Each iteration performs:
//!   * E-step: accumulate fractional ("expected") alignment counts;
//!   * M-step: normalize those counts into updated `t(f | e)` values.

use crate::corpus::SentencePair;
use crate::model::IBMModel1;
use crate::vocab::TokenId;
use std::collections::HashMap;

/// Per-iteration diagnostics.
#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub iteration: usize,
    /// Length-normalized IBM Model 1 log-likelihood per target token,
    /// evaluated with the parameters produced by this iteration. EM
    /// guarantees this never decreases from one iteration to the next, which
    /// makes it a convenient convergence check in class.
    pub avg_log_likelihood_per_token: f64,
}

/// Run `iterations` rounds of EM, mutating `model` in place.
///
/// Returns one [`TrainingReport`] per iteration.
pub fn train(
    model: &mut IBMModel1,
    corpus: &[SentencePair],
    iterations: usize,
) -> Vec<TrainingReport> {
    let mut reports = Vec::with_capacity(iterations);

    for iteration in 1..=iterations {
        let mut count: HashMap<TokenId, HashMap<TokenId, f64>> = HashMap::new();
        let mut total: HashMap<TokenId, f64> = HashMap::new();

        // ---- E-step: expected counts -------------------------------------
        for pair in corpus {
            for &f in &pair.target {
                let denominator: f64 =
                    pair.source.iter().map(|&e| model.probability(e, f)).sum();

                if denominator == 0.0 {
                    continue;
                }

                for &e in &pair.source {
                    let delta = model.probability(e, f) / denominator;
                    *count.entry(e).or_default().entry(f).or_insert(0.0) += delta;
                    *total.entry(e).or_insert(0.0) += delta;
                }
            }
        }

        // ---- M-step: normalize to update t(f | e) ------------------------
        model.table.clear();
        for (e, f_counts) in count {
            let total_for_e = total.get(&e).copied().unwrap_or(0.0);
            if total_for_e == 0.0 {
                continue;
            }
            for (f, c) in f_counts {
                model.set_probability(e, f, c / total_for_e);
            }
        }

        reports.push(TrainingReport {
            iteration,
            avg_log_likelihood_per_token: avg_log_likelihood_per_token(model, corpus),
        });
    }

    reports
}

/// Average, over all target tokens in the corpus, of the IBM Model 1
/// per-token log-likelihood contribution.
///
/// The full IBM Model 1 sentence likelihood is
///
/// ```text
/// Pr(f | e) = (epsilon / (l + 1)^m) * prod_j  sum_i t(f_j | e_i)
/// ```
///
/// We drop the constant `epsilon` (it does not affect the *shape* of the
/// curve) and fold the `1 / (l + 1)` factor into each per-token term, giving
///
/// ```text
/// per_token_j = ln( (1 / (l + 1)) * sum_i t(f_j | e_i) )
/// ```
///
/// where `l + 1` is the source length *including* `<NULL>` (which is already
/// present in `pair.source`). Summing these over all target tokens and
/// dividing by the token count yields a properly averaged, monotonically
/// improving quantity — unlike the original code, this is a real (length-
/// normalized) log-likelihood term rather than an unlabeled proxy.
fn avg_log_likelihood_per_token(model: &IBMModel1, corpus: &[SentencePair]) -> f64 {
    let mut log_sum = 0.0;
    let mut token_count = 0usize;

    for pair in corpus {
        // pair.source already contains <NULL>, so its length is (l + 1).
        let normalizer = pair.source.len() as f64;

        for &f in &pair.target {
            let inner: f64 = pair.source.iter().map(|&e| model.probability(e, f)).sum();
            if inner > 0.0 {
                log_sum += (inner / normalizer).ln();
                token_count += 1;
            }
        }
    }

    if token_count == 0 {
        0.0
    } else {
        log_sum / token_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::SentencePair;

    fn controlled_corpus() -> Vec<SentencePair> {
        // <NULL>=0, "the"=1, "house"=2, "book"=3
        // "la"=10, "maison"=11, "le"=12, "livre"=13
        // Two sentences share "the"/<NULL> but differ in content words, which
        // breaks the symmetry and forces EM to move the probabilities.
        vec![
            SentencePair {
                source: vec![0, 1, 2], // <NULL> the house
                target: vec![10, 11],  // la maison
            },
            SentencePair {
                source: vec![0, 1, 3], // <NULL> the book
                target: vec![12, 13],  // le livre
            },
        ]
    }

    #[test]
    fn probabilities_change_during_training() {
        // Note: with a uniform co-occurrence initialization a single EM step
        // can reproduce the initial table on highly symmetric corpora (it is
        // briefly a fixed point). Over several iterations the asymmetry
        // between content words compounds and the table must move. We assert
        // the realistic, multi-iteration behavior.
        let corpus = vec![
            // <NULL>=0 the=1 house=2 ; la=10 maison=11
            SentencePair { source: vec![0, 1, 2], target: vec![10, 11] },
            // <NULL>=0 a=3 house=2 ; une=12 maison=11  -> "house"/"maison" reinforced
            SentencePair { source: vec![0, 3, 2], target: vec![12, 11] },
        ];
        let mut model = IBMModel1::new_uniform_from_corpus(&corpus);
        let before = model.probability(2, 11); // t(maison | house)
        train(&mut model, &corpus, 10);
        let after = model.probability(2, 11);
        assert!(
            after > before,
            "t(maison|house) should increase: {before} -> {after}"
        );
    }

    #[test]
    fn rows_stay_normalized_after_training() {
        let corpus = controlled_corpus();
        let mut model = IBMModel1::new_uniform_from_corpus(&corpus);
        train(&mut model, &corpus, 5);
        for &e in &[0usize, 1, 2, 3] {
            if model.table.contains_key(&e) {
                assert!((model.row_sum(e) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn log_likelihood_is_monotonically_non_decreasing() {
        // EM is guaranteed not to decrease the likelihood.
        let corpus = controlled_corpus();
        let mut model = IBMModel1::new_uniform_from_corpus(&corpus);
        let reports = train(&mut model, &corpus, 8);
        for w in reports.windows(2) {
            assert!(
                w[1].avg_log_likelihood_per_token + 1e-9
                    >= w[0].avg_log_likelihood_per_token,
                "likelihood decreased between iterations"
            );
        }
    }
}
