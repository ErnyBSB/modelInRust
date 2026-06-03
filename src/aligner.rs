//! Word alignment from a trained model (spec, section 9).
//!
//! For each target word `f_j`, pick the source word `e_i` with the highest
//! `t(f_j | e_i)`. This is the most probable word-to-word link under the
//! learned table — it is *not* a translation.

use crate::corpus::SentencePair;
use crate::model::IBMModel1;
use crate::vocab::TokenId;

/// A single most-probable link from a target word to a source word.
#[derive(Debug, Clone)]
pub struct AlignmentLink {
    pub source: TokenId,
    pub target: TokenId,
    pub score: f64,
}

/// Produce Viterbi-style alignments for one sentence pair.
///
/// `null_id` is the ID of the `<NULL>` token. When `allow_null` is `false`,
/// `<NULL>` is still considered as a fallback only if no real source word has
/// positive probability, so a target word never silently aligns to nothing
/// when a real candidate exists.
pub fn align(
    model: &IBMModel1,
    pair: &SentencePair,
    null_id: TokenId,
    allow_null: bool,
) -> Vec<AlignmentLink> {
    let mut links = Vec::with_capacity(pair.target.len());

    for &f in &pair.target {
        let mut best_source = pair.source[0];
        let mut best_score = f64::NEG_INFINITY;

        for &e in &pair.source {
            if !allow_null && e == null_id {
                continue;
            }
            let score = model.probability(e, f);
            if score > best_score {
                best_score = score;
                best_source = e;
            }
        }

        // Fallback: if every real word scored 0 (or NULL was skipped and only
        // NULL existed), allow NULL so we always emit a link.
        if best_score <= 0.0 {
            best_source = null_id;
            best_score = model.probability(null_id, f);
        }

        links.push(AlignmentLink {
            source: best_source,
            target: f,
            score: best_score,
        });
    }

    links
}

/// Render alignments in the widely used Pharaoh format (`t-s` index pairs,
/// 0-based, counting real source positions only — `<NULL>` links are
/// dropped, as is conventional). Implements a suggested extension.
pub fn to_pharaoh(links: &[AlignmentLink], pair: &SentencePair, null_id: TokenId) -> String {
    // Map each source token id to its (NULL-excluded) position.
    let mut positions = Vec::new();
    let mut real_index = 0usize;
    for &e in &pair.source {
        if e == null_id {
            positions.push(None);
        } else {
            positions.push(Some(real_index));
            real_index += 1;
        }
    }

    let mut out = Vec::new();
    for (t_idx, link) in links.iter().enumerate() {
        if link.source == null_id {
            continue;
        }
        if let Some(src_pos) = pair
            .source
            .iter()
            .position(|&e| e == link.source)
            .and_then(|i| positions[i])
        {
            out.push(format!("{t_idx}-{src_pos}"));
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::IBMModel1;

    #[test]
    fn aligns_target_to_highest_probability_source() {
        // <NULL>=0, the=1, house=2 ; la=10, maison=11
        let pair = SentencePair {
            source: vec![0, 1, 2],
            target: vec![11], // maison
        };
        let mut model = IBMModel1::default();
        model.set_probability(0, 11, 0.05);
        model.set_probability(1, 11, 0.10);
        model.set_probability(2, 11, 0.85); // house -> maison wins

        let links = align(&model, &pair, 0, true);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].source, 2); // house
    }

    #[test]
    fn pharaoh_skips_null_links() {
        let pair = SentencePair {
            source: vec![0, 1],
            target: vec![10],
        };
        let mut model = IBMModel1::default();
        model.set_probability(0, 10, 0.9); // best link is NULL
        let links = align(&model, &pair, 0, true);
        assert_eq!(to_pharaoh(&links, &pair, 0), "");
    }
}
