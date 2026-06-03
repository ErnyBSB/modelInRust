use crate::vocab::{TokenId, Vocabulary, NULL_TOKEN};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Debug, Clone)]
pub struct SentencePair {
    /// Source sentence, corresponding to e in IBM Model 1.
    /// It includes the special <NULL> token at position 0.
    pub source: Vec<TokenId>,

    /// Target sentence, corresponding to f in IBM Model 1.
    pub target: Vec<TokenId>,
}

pub fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .to_lowercase()
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

pub fn load_tsv(
    path: &str,
    source_vocab: &mut Vocabulary,
    target_vocab: &mut Vocabulary,
) -> io::Result<Vec<SentencePair>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let null_id = source_vocab.get_or_insert(NULL_TOKEN);
    let mut corpus = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 2 {
            eprintln!(
                "Skipping line {}: expected exactly two tab-separated columns.",
                line_no + 1
            );
            continue;
        }

        let mut source_ids = vec![null_id];
        source_ids.extend(
            tokenize(parts[0])
                .iter()
                .map(|word| source_vocab.get_or_insert(word)),
        );

        let target_ids = tokenize(parts[1])
            .iter()
            .map(|word| target_vocab.get_or_insert(word))
            .collect::<Vec<_>>();

        if !source_ids.is_empty() && !target_ids.is_empty() {
            corpus.push(SentencePair {
                source: source_ids,
                target: target_ids,
            });
        }
    }

    Ok(corpus)
}
