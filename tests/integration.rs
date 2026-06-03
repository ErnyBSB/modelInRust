//! End-to-end integration test.
//!
//! Reproduces the spec's expectation (section 13, item 5): after training on a
//! controlled corpus, "maison" should align with "house" and "livre" with
//! "book". Because the binary crate exposes its modules privately, we re-run
//! the same logic here against the public surface re-exported through a thin
//! library shim is not available; instead we duplicate the minimal pipeline
//! using the crate's modules via `path` includes.
//!
//! To keep the project a single binary crate (as in the original), this test
//! drives the program through its data file using the same algorithm, by
//! pulling the modules in directly.

#[path = "../src/vocab.rs"]
mod vocab;
#[path = "../src/corpus.rs"]
mod corpus;
#[path = "../src/model.rs"]
mod model;
#[path = "../src/trainer.rs"]
mod trainer;
#[path = "../src/aligner.rs"]
mod aligner;

use aligner::align;
use model::IBMModel1;
use trainer::train;
use vocab::{Vocabulary, NULL_TOKEN};

fn write_temp_corpus() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push("ibm_model_1_integration_corpus.tsv");
    std::fs::write(
        &path,
        "the house\tla maison\n\
         the book\tle livre\n\
         a house\tune maison\n\
         my book\tmon livre\n\
         the small house\tla petite maison\n\
         the small book\tle petit livre\n",
    )
    .unwrap();
    path
}

#[test]
fn maison_aligns_with_house_and_livre_with_book() {
    let path = write_temp_corpus();

    let mut source_vocab = Vocabulary::new();
    source_vocab.get_or_insert(NULL_TOKEN);
    let mut target_vocab = Vocabulary::new();

    let (corpus, _stats) = corpus::load_tsv(
        path.to_str().unwrap(),
        &mut source_vocab,
        &mut target_vocab,
    )
    .unwrap();

    let null_id = source_vocab.id(NULL_TOKEN).unwrap();

    let mut m = IBMModel1::new_uniform_from_corpus(&corpus);
    train(&mut m, &corpus, 30);

    let house = source_vocab.id("house").unwrap();
    let maison = target_vocab.id("maison").unwrap();
    let book = source_vocab.id("book").unwrap();
    let livre = target_vocab.id("livre").unwrap();

    // house should be the best source for maison among all sources.
    assert!(
        m.probability(house, maison) > m.probability(null_id, maison),
        "house should beat NULL for maison"
    );
    assert!(
        m.probability(book, livre) > m.probability(null_id, livre),
        "book should beat NULL for livre"
    );

    // And the aligner should pick those links on a concrete sentence.
    let pair = &corpus[0]; // "the house" / "la maison"
    let links = align(&m, pair, null_id, false);
    let maison_link = links.iter().find(|l| l.target == maison).unwrap();
    assert_eq!(
        maison_link.source, house,
        "maison should align to house after training"
    );

    let _ = std::fs::remove_file(path);
}
