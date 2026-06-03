//! Vocabulary: a bidirectional mapping between words (`String`) and dense
//! integer token IDs (`TokenId`).
//!
//! Using integer IDs instead of strings inside the model keeps indexing cheap
//! and mirrors how real NLP pipelines represent tokens (spec, section 3).

use std::collections::HashMap;

/// A dense integer identifier for a token. IDs are assigned sequentially
/// starting at `0` in insertion order.
pub type TokenId = usize;

/// The special "empty cept" / NULL source token, written `e0` in the IBM
/// Model 1 literature. It lets a target word be generated without any real
/// source-language counterpart (spec, section 4).
pub const NULL_TOKEN: &str = "<NULL>";

/// Bidirectional word <-> id map.
#[derive(Debug, Default)]
pub struct Vocabulary {
    word_to_id: HashMap<String, TokenId>,
    id_to_word: Vec<String>,
}

#[allow(dead_code)] // public API; some methods exercised only by tests/consumers
impl Vocabulary {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the ID for `word`, inserting it if it is not present yet.
    pub fn get_or_insert(&mut self, word: &str) -> TokenId {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }

        let id = self.id_to_word.len();
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.push(word.to_string());
        id
    }

    /// Look up the ID of a word without inserting it.
    pub fn id(&self, word: &str) -> Option<TokenId> {
        self.word_to_id.get(word).copied()
    }

    /// Return `true` if `word` is already known to the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.word_to_id.contains_key(word)
    }

    /// Resolve a token ID back to its word. Unknown IDs return a sentinel
    /// string rather than panicking, which keeps printing code simple.
    pub fn word(&self, id: TokenId) -> &str {
        self.id_to_word
            .get(id)
            .map(String::as_str)
            .unwrap_or("<UNKNOWN_ID>")
    }

    /// Number of distinct tokens stored.
    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }

    /// Whether the vocabulary holds no tokens. Provided because Clippy
    /// (correctly) flags any type exposing `len` without `is_empty`.
    pub fn is_empty(&self) -> bool {
        self.id_to_word.is_empty()
    }
}
