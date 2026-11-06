//! Count and limit text tokens for a model. Adapted from Mamba AgentKit.

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;

/// A tokenizer for a model supported by `tiktoken-rs`.
pub struct TiktokenTokenizer {
    bpe: CoreBPE,
}

impl TiktokenTokenizer {
    /// Load the encoding for a model name.
    pub fn for_model(model: &str) -> Result<Self> {
        let bpe = tiktoken_rs::get_bpe_from_model(model)
            .with_context(|| format!("Failed to load tokenizer for model {model:?}"))?;
        Ok(Self { bpe })
    }

    /// Load the `cl100k_base` encoding directly.
    pub fn cl100k_base() -> Result<Self> {
        let bpe = tiktoken_rs::cl100k_base().context("Failed to load cl100k_base tokenizer")?;
        Ok(Self { bpe })
    }

    /// Load the `o200k_base` encoding directly.
    pub fn o200k_base() -> Result<Self> {
        let bpe = tiktoken_rs::o200k_base().context("Failed to load o200k_base tokenizer")?;
        Ok(Self { bpe })
    }

    /// Count model tokens, including any special tokens in the input.
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    /// Return a valid UTF-8 prefix that fits within the token limit.
    pub fn truncate(&self, text: &str, max_tokens: usize) -> String {
        let tokens = self.bpe.encode_with_special_tokens(text);
        if tokens.len() <= max_tokens {
            return text.to_owned();
        }

        for end in (1..=max_tokens).rev() {
            // A token boundary can fall inside a character. The shortened text can also
            // tokenize differently when it ends there.
            if let Ok(prefix) = self.bpe.decode(tokens[..end].to_vec())
                && self.count(&prefix) <= max_tokens
            {
                return prefix;
            }
        }

        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_limits_text_without_splitting_a_character() -> Result<()> {
        let tokenizer = TiktokenTokenizer::cl100k_base()?;
        let text = "hello🙂";
        assert_eq!(tokenizer.count(text), 3);
        assert_eq!(tokenizer.truncate(text, 2), "hello");
        assert_eq!(tokenizer.truncate(text, 3), text);
        assert_eq!(tokenizer.truncate(text, 0), "");
        Ok(())
    }

    #[test]
    fn reports_an_unknown_model() {
        assert!(TiktokenTokenizer::for_model("faberline-unknown-model").is_err());
    }
}
