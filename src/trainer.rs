use crate::corpus::SentencePair;
use crate::model::IBMModel1;
use crate::vocab::TokenId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub iteration: usize,
    pub average_log_likelihood: f64,
}

pub fn train(model: &mut IBMModel1, corpus: &[SentencePair], iterations: usize) -> Vec<TrainingReport> {
    let mut reports = Vec::new();

    for iteration in 1..=iterations {
        let mut count: HashMap<TokenId, HashMap<TokenId, f64>> = HashMap::new();
        let mut total: HashMap<TokenId, f64> = HashMap::new();

        // E-step: compute expected counts.
        for pair in corpus {
            for &f in &pair.target {
                let denominator: f64 = pair
                    .source
                    .iter()
                    .map(|&e| model.probability(e, f))
                    .sum();

                if denominator == 0.0 {
                    continue;
                }

                for &e in &pair.source {
                    let delta = model.probability(e, f) / denominator;

                    *count
                        .entry(e)
                        .or_default()
                        .entry(f)
                        .or_insert(0.0) += delta;

                    *total.entry(e).or_insert(0.0) += delta;
                }
            }
        }

        // M-step: normalize expected counts to update t(f | e).
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
            average_log_likelihood: average_log_likelihood(model, corpus),
        });
    }

    reports
}

/// A didactic likelihood proxy based on IBM Model 1, omitting the constant epsilon.
/// For each target word f_j, we compute: sum_i t(f_j | e_i) / len(source).
fn average_log_likelihood(model: &IBMModel1, corpus: &[SentencePair]) -> f64 {
    let mut log_sum = 0.0;
    let mut target_token_count = 0usize;

    for pair in corpus {
        let source_len = pair.source.len() as f64;

        for &f in &pair.target {
            let denominator: f64 = pair
                .source
                .iter()
                .map(|&e| model.probability(e, f))
                .sum();

            if denominator > 0.0 {
                log_sum += (denominator / source_len).ln();
                target_token_count += 1;
            }
        }
    }

    if target_token_count == 0 {
        0.0
    } else {
        log_sum / target_token_count as f64
    }
}
