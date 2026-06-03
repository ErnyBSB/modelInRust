use crate::corpus::SentencePair;
use crate::model::IBMModel1;
use crate::vocab::TokenId;

#[derive(Debug, Clone)]
pub struct AlignmentLink {
    pub source: TokenId,
    pub target: TokenId,
    pub score: f64,
}

pub fn align(model: &IBMModel1, pair: &SentencePair) -> Vec<AlignmentLink> {
    let mut links = Vec::new();

    for &f in &pair.target {
        let mut best_source = pair.source[0];
        let mut best_score = f64::NEG_INFINITY;

        for &e in &pair.source {
            let score = model.probability(e, f);
            if score > best_score {
                best_score = score;
                best_source = e;
            }
        }

        links.push(AlignmentLink {
            source: best_source,
            target: f,
            score: best_score,
        });
    }

    links
}
