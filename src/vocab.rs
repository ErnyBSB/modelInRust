use std::collections::HashMap;

pub type TokenId = usize;

pub const NULL_TOKEN: &str = "<NULL>";

#[derive(Debug, Default)]
pub struct Vocabulary {
    word_to_id: HashMap<String, TokenId>,
    id_to_word: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        }
    }

    pub fn get_or_insert(&mut self, word: &str) -> TokenId {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }

        let id = self.id_to_word.len();
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.push(word.to_string());
        id
    }

    pub fn word(&self, id: TokenId) -> &str {
        self.id_to_word
            .get(id)
            .map(|s| s.as_str())
            .unwrap_or("<UNKNOWN_ID>")
    }

    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }
}
