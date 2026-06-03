//! Orchestration and CLI for the didactic IBM Model 1 implementation.

mod aligner;
mod corpus;
mod model;
mod trainer;
mod vocab;

use crate::aligner::{align, to_pharaoh};
use crate::corpus::load_tsv;
use crate::model::IBMModel1;
use crate::trainer::train;
use crate::vocab::{Vocabulary, NULL_TOKEN};
use std::env;
use std::fs;
use std::process;

#[derive(Debug)]
struct Config {
    corpus_path: String,
    iterations: usize,
    top: usize,
    examples: usize,
    save_json: Option<String>,
    pharaoh: bool,
    allow_null_alignment: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            corpus_path: "data/toy.tsv".to_string(),
            iterations: 10,
            top: 20,
            examples: 3,
            save_json: None,
            pharaoh: false,
            allow_null_alignment: true,
        }
    }
}

fn main() {
    let config = match parse_args(env::args().skip(1).collect()) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("error: {msg}\n");
            print_help();
            process::exit(2);
        }
    };

    let mut source_vocab = Vocabulary::new();
    source_vocab.get_or_insert(NULL_TOKEN); // guarantee NULL has id 0
    let mut target_vocab = Vocabulary::new();

    let (corpus, stats) = match load_tsv(&config.corpus_path, &mut source_vocab, &mut target_vocab)
    {
        Ok(result) => result,
        Err(err) => {
            eprintln!("error: could not load corpus '{}': {err}", config.corpus_path);
            process::exit(1);
        }
    };

    if corpus.is_empty() {
        eprintln!(
            "error: the corpus is empty. Provide a TSV file with 'source<TAB>target' pairs."
        );
        process::exit(1);
    }

    let null_id = source_vocab.id(NULL_TOKEN).expect("NULL token must exist");

    println!("IBM Model 1 - didactic Rust implementation");
    println!("Corpus: {}", config.corpus_path);
    println!("Sentence pairs: {} (skipped lines: {})", corpus.len(), stats.lines_skipped);
    println!("Source vocabulary size: {}", source_vocab.len());
    println!("Target vocabulary size: {}", target_vocab.len());
    println!("Iterations: {}", config.iterations);
    println!();

    let mut model = IBMModel1::new_uniform_from_corpus(&corpus);
    let reports = train(&mut model, &corpus, config.iterations);

    println!("Training report (log-likelihood per token should not decrease)");
    for report in &reports {
        println!(
            "  iteration {:>2}: avg log-likelihood/token = {:.6}",
            report.iteration, report.avg_log_likelihood_per_token
        );
    }
    println!();

    println!("Top learned translation probabilities t(f | e)");
    for (rank, (e, f, probability)) in model
        .sorted_probabilities()
        .into_iter()
        .take(config.top)
        .enumerate()
    {
        println!(
            "  {:>2}. t({:<10} | {:<10}) = {:.6}",
            rank + 1,
            target_vocab.word(f),
            source_vocab.word(e),
            probability
        );
    }
    println!();

    println!("Example alignments");
    for (idx, pair) in corpus.iter().take(config.examples).enumerate() {
        println!("Sentence pair {}", idx + 1);
        let links = align(&model, pair, null_id, config.allow_null_alignment);
        for link in &links {
            println!(
                "  {:<10} -> {:<10} score={:.6}",
                target_vocab.word(link.target),
                source_vocab.word(link.source),
                link.score
            );
        }
        if config.pharaoh {
            println!("  pharaoh: {}", to_pharaoh(&links, pair, null_id));
        }
        println!();
    }

    if let Some(path) = &config.save_json {
        let json = model.to_named_json(&source_vocab, &target_vocab);
        match fs::write(path, json) {
            Ok(()) => println!("Saved learned translation table to '{path}'."),
            Err(err) => eprintln!("warning: could not write '{path}': {err}"),
        }
    }
}

/// Parse CLI arguments strictly: an unknown flag, a missing value, or an
/// unparsable number is a hard error rather than being silently ignored.
fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut i = 0;

    // Small helper to fetch the value following a flag.
    fn value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
        *i += 1;
        args.get(*i)
            .map(String::as_str)
            .ok_or_else(|| format!("missing value after {flag}"))
    }

    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => config.corpus_path = value(&args, &mut i, "--corpus")?.to_string(),
            "--iterations" => {
                let v = value(&args, &mut i, "--iterations")?;
                config.iterations = v
                    .parse()
                    .map_err(|_| format!("--iterations expects a non-negative integer, got '{v}'"))?;
            }
            "--top" => {
                let v = value(&args, &mut i, "--top")?;
                config.top = v
                    .parse()
                    .map_err(|_| format!("--top expects a non-negative integer, got '{v}'"))?;
            }
            "--examples" => {
                let v = value(&args, &mut i, "--examples")?;
                config.examples = v
                    .parse()
                    .map_err(|_| format!("--examples expects a non-negative integer, got '{v}'"))?;
            }
            "--save-json" => config.save_json = Some(value(&args, &mut i, "--save-json")?.to_string()),
            "--pharaoh" => config.pharaoh = true,
            "--no-null-align" => config.allow_null_alignment = false,
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("Usage:");
    println!("  cargo run -- [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --corpus <PATH>       TSV file: source<TAB>target  (default: data/toy.tsv)");
    println!("  --iterations <N>      Number of EM iterations       (default: 10)");
    println!("  --top <N>             Learned probabilities to print (default: 20)");
    println!("  --examples <N>        Example sentence alignments     (default: 3)");
    println!("  --save-json <PATH>    Write learned t(f|e) table as JSON");
    println!("  --pharaoh             Also print alignments in Pharaoh format");
    println!("  --no-null-align       Prefer real source words over <NULL> in alignment");
    println!("  -h, --help            Show this help and exit");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_when_no_args() {
        let cfg = parse_args(vec![]).unwrap();
        assert_eq!(cfg.iterations, 10);
        assert_eq!(cfg.corpus_path, "data/toy.tsv");
    }

    #[test]
    fn unknown_flag_is_error() {
        assert!(parse_args(vec!["--nope".into()]).is_err());
    }

    #[test]
    fn invalid_number_is_error_not_silent_default() {
        let err = parse_args(vec!["--iterations".into(), "abc".into()]).unwrap_err();
        assert!(err.contains("--iterations"));
    }

    #[test]
    fn missing_value_is_error() {
        assert!(parse_args(vec!["--corpus".into()]).is_err());
    }
}
