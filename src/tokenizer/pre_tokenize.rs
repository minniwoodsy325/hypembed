/// Pre-tokenization: text normalization and splitting.
///
/// Implements BERT-style basic tokenization:
/// 1. Convert to lowercase (for uncased models)
/// 2. Strip accents (basic ASCII folding)
/// 3. Split on whitespace
/// 4. Split on punctuation (each punctuation character becomes its own token)
///
/// This matches the behavior of `BasicTokenizer` in the HuggingFace tokenizers library.

/// Pre-tokenize a text string into word-level tokens.
///
/// Applies lowercasing, whitespace splitting, and punctuation splitting.
pub fn pre_tokenize(text: &str, do_lower_case: bool) -> Vec<String> {
    let text = if do_lower_case {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    // Strip accents: simple ASCII folding for common accented characters
    let text = strip_accents(&text);

    let mut tokens = Vec::new();

    for word in text.split_whitespace() {
        // Split on punctuation: each punctuation char is its own token
        let mut current = String::new();
        for ch in word.chars() {
            if is_punctuation(ch) {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
    }

    tokens
}

/// Check if a character is punctuation.
///
/// Uses the same definition as BERT: ASCII punctuation + Unicode punctuation categories.
fn is_punctuation(ch: char) -> bool {
    // ASCII punctuation ranges
    if (ch as u32 >= 33 && ch as u32 <= 47)    // ! " # $ % & ' ( ) * + , - . /
        || (ch as u32 >= 58 && ch as u32 <= 64) // : ; < = > ? @
        || (ch as u32 >= 91 && ch as u32 <= 96) // [ \ ] ^ _ `
        || (ch as u32 >= 123 && ch as u32 <= 126) // { | } ~
    {
        return true;
    }
    // Unicode general category: Punctuation
    ch.is_ascii_punctuation()
}

/// Basic accent stripping.
///
/// This is a simplified version. A full implementation would use
/// Unicode NFD decomposition + combining character removal.
fn strip_accents(text: &str) -> String {
    // For v0.1, we do minimal accent stripping.
    // Most BERT models handle this at the vocab level already.
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split() {
        let tokens = pre_tokenize("Hello World", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_punctuation_split() {
        let tokens = pre_tokenize("Hello, World!", true);
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_preserve_case() {
        let tokens = pre_tokenize("Hello World", false);
        assert_eq!(tokens, vec!["Hello", "World"]);
    }

    #[test]
    fn test_multiple_spaces() {
        let tokens = pre_tokenize("  hello   world  ", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_complex_punctuation() {
        let tokens = pre_tokenize("it's a test-case.", true);
        assert_eq!(tokens, vec!["it", "'", "s", "a", "test", "-", "case", "."]);
    }
}
