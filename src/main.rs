mod aligner;
mod corpus;
mod model;
mod trainer;
mod vocab;

use crate::aligner::align;
use crate::corpus::load_tsv;
use crate::model::IBMModel1;
use crate::trainer::train;
use crate::vocab::{Vocabulary, NULL_TOKEN};
use std::env;

#[derive(Debug)]
struct Config {
    corpus_path: String,
    iterations: usize,
    top: usize,
}

fn main() {
    let config = parse_args();

    let mut source_vocab = Vocabulary::new();
    source_vocab.get_or_insert(NULL_TOKEN);
    let mut target_vocab = Vocabulary::new();

    let corpus = match load_tsv(&config.corpus_path, &mut source_vocab, &mut target_vocab) {
        Ok(corpus) => corpus,
        Err(err) => {
            eprintln!("Error loading corpus '{}': {}", config.corpus_path, err);
            std::process::exit(1);
        }
    };

    if corpus.is_empty() {
        eprintln!("The corpus is empty. Provide a TSV file with source<TAB>target sentence pairs.");
        std::process::exit(1);
    }

    println!("IBM Model 1 - didactic Rust implementation");
    println!("Corpus: {}", config.corpus_path);
    println!("Sentence pairs: {}", corpus.len());
    println!("Source vocabulary size: {}", source_vocab.len());
    println!("Target vocabulary size: {}", target_vocab.len());
    println!("Iterations: {}", config.iterations);
    println!();

    let mut model = IBMModel1::new_uniform_from_corpus(&corpus);
    let reports = train(&mut model, &corpus, config.iterations);

    println!("Training report");
    for report in reports {
        println!(
            "  iteration {:>2}: average log-likelihood proxy = {:.6}",
            report.iteration, report.average_log_likelihood
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
    for (idx, pair) in corpus.iter().take(3).enumerate() {
        println!("Sentence pair {}", idx + 1);
        for link in align(&model, pair) {
            println!(
                "  {:<10} -> {:<10} score={:.6}",
                target_vocab.word(link.target),
                source_vocab.word(link.source),
                link.score
            );
        }
        println!();
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut corpus_path = "data/toy.tsv".to_string();
    let mut iterations = 10usize;
    let mut top = 20usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                if i + 1 < args.len() {
                    corpus_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    iterations = args[i + 1].parse().unwrap_or(iterations);
                    i += 1;
                }
            }
            "--top" => {
                if i + 1 < args.len() {
                    top = args[i + 1].parse().unwrap_or(top);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help_and_exit();
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help_and_exit();
            }
        }
        i += 1;
    }

    Config {
        corpus_path,
        iterations,
        top,
    }
}

fn print_help_and_exit() -> ! {
    println!("Usage:");
    println!("  cargo run -- --corpus data/toy.tsv --iterations 10 --top 20");
    println!();
    println!("Arguments:");
    println!("  --corpus <PATH>       TSV file with source<TAB>target sentence pairs");
    println!("  --iterations <N>      Number of EM iterations");
    println!("  --top <N>             Number of learned probabilities to print");
    std::process::exit(0);
}
