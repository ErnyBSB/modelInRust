//! Corpus loading and tokenization.
//!
//! Reads a UTF-8 TSV file with two tab-separated columns
//! (`source<TAB>target`), tokenizes both sides, and prepends the `<NULL>`
//! token to every source sentence (spec, sections 4 and 10).

use crate::vocab::{TokenId, Vocabulary, NULL_TOKEN};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

/// A single aligned sentence pair.
#[derive(Debug, Clone)]
pub struct SentencePair {
    /// Source sentence (`e` in the IBM Model 1 notation).
    /// Always includes the `<NULL>` token at position 0.
    pub source: Vec<TokenId>,

    /// Target sentence (`f` in the IBM Model 1 notation).
    pub target: Vec<TokenId>,
}

/// Summary statistics returned alongside the parsed corpus so the caller can
/// report how clean the input was.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoadStats {
    pub pairs_loaded: usize,
    pub lines_skipped: usize,
}

/// Lowercase and split a sentence into word tokens.
///
/// Punctuation is trimmed from each token edge. Unlike the original version,
/// this trims *Unicode* punctuation (via `char::is_alphanumeric`) so French
/// guillemets, em dashes, etc. are handled, not just ASCII punctuation.
/// Combining accents inside a word (e.g. `petite`, `maison`) are preserved.
pub fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

/// Load a TSV corpus, populating the two vocabularies as a side effect.
///
/// Blank lines and lines beginning with `#` are treated as comments. Lines
/// that do not have exactly two tab-separated columns are skipped with a
/// warning and counted in [`LoadStats::lines_skipped`].
pub fn load_tsv(
    path: &str,
    source_vocab: &mut Vocabulary,
    target_vocab: &mut Vocabulary,
) -> io::Result<(Vec<SentencePair>, LoadStats)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let null_id = source_vocab.get_or_insert(NULL_TOKEN);

    let mut corpus = Vec::new();
    let mut stats = LoadStats::default();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() != 2 {
            eprintln!(
                "warning: skipping line {}: expected exactly two tab-separated columns",
                line_no + 1
            );
            stats.lines_skipped += 1;
            continue;
        }

        let mut source_ids = vec![null_id];
        source_ids.extend(
            tokenize(parts[0])
                .iter()
                .map(|word| source_vocab.get_or_insert(word)),
        );

        let target_ids: Vec<TokenId> = tokenize(parts[1])
            .iter()
            .map(|word| target_vocab.get_or_insert(word))
            .collect();

        // A source sentence always has at least <NULL>; require a real target.
        if source_ids.len() > 1 && !target_ids.is_empty() {
            corpus.push(SentencePair {
                source: source_ids,
                target: target_ids,
            });
            stats.pairs_loaded += 1;
        } else {
            eprintln!(
                "warning: skipping line {}: empty source or target after tokenization",
                line_no + 1
            );
            stats.lines_skipped += 1;
        }
    }

    Ok((corpus, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_and_strips_edge_punctuation() {
        assert_eq!(tokenize("The House."), vec!["the", "house"]);
        assert_eq!(tokenize("  Le   livre !"), vec!["le", "livre"]);
    }

    #[test]
    fn tokenize_preserves_internal_accents() {
        assert_eq!(tokenize("petite maison"), vec!["petite", "maison"]);
    }

    #[test]
    fn tokenize_drops_pure_punctuation_tokens() {
        assert_eq!(tokenize("a -- b"), vec!["a", "b"]);
    }
}
