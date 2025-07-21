# --- text_preprocessing.py ---
"""
Text preprocessing optimized for authorship attribution.
Preserves stylistic signals while normalizing text for feature extraction.
"""
import re
import string
import pandas as pd
import numpy as np

try:
    import spacy

    # Try to load spaCy model with GPU support
    try:
        # Load with GPU support if available
        nlp = spacy.load('en_core_web_sm')

        # Configure for GPU if available
        if spacy.prefer_gpu():
            print("🚀 spaCy GPU acceleration enabled!")
        else:
            print("⚡ spaCy CPU mode (GPU not available)")

        # Optimize pipeline for speed
        nlp.disable_pipes(['ner', 'parser'])  # Keep only tokenizer and tagger
        print("✅ spaCy loaded and optimized for authorship attribution")

        SPACY_AVAILABLE = True

    except OSError as e:
        print(f"⚠️  spaCy model not found: {e}")
        print("📦 Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False

except ImportError:
    print("⚠️  spaCy not available. Using basic preprocessing.")
    print("📦 Install with: pip install spacy")
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download required NLTK data (comprehensive list for all NLTK versions)
    required_data = [
        'stopwords',
        'punkt',
        'punkt_tab',  # New tokenizer data
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',  # English-specific tagger
        'omw-1.4',  # Open Multilingual Wordnet
        'brown',  # Brown corpus (sometimes needed)
        'universal_tagset',  # Universal POS tagset
        'vader_lexicon',  # Sentiment analysis (optional)
    ]

    print("📦 Downloading required NLTK data (this may take a moment)...")
    download_success = {}

    for data_name in required_data:
        try:
            result = nltk.download(data_name, quiet=False)
            download_success[data_name] = result
            if result:
                print(f"   ✅ Downloaded {data_name}")
            else:
                print(f"   ✓ {data_name} already available")
        except Exception as e:
            print(f"   ⚠️  Could not download {data_name}: {e}")
            download_success[data_name] = False

    # Test if critical resources are available
    try:
        # Test tokenization
        nltk.word_tokenize("test sentence")
        print("   ✅ NLTK tokenization working")

        # Test POS tagging
        tokens = nltk.word_tokenize("test sentence")
        nltk.pos_tag(tokens)
        print("   ✅ NLTK POS tagging working")

    except Exception as e:
        print(f"   ⚠️  NLTK functionality test failed: {e}")

    NLTK_AVAILABLE = True
    lemmatizer = WordNetLemmatizer()

except ImportError:
    print("⚠️  NLTK not available. Using basic preprocessing.")
    NLTK_AVAILABLE = False


def preprocess_dataframe(df):
    """
    Apply preprocessing to text column of a DataFrame.

    Args:
        df: DataFrame with 'text' column

    Returns:
        DataFrame with additional preprocessing columns
    """
    print(f"🔧 Preprocessing {len(df):,} text samples for authorship attribution...")

    # Create copy to avoid modifying original
    processed_df = df.copy()

    # Apply preprocessing
    processed_df['text_clean'] = processed_df['text'].apply(preprocess_text)
    processed_df['text_tokens'] = processed_df['text'].apply(tokenize_text)
    processed_df['text_for_tfidf'] = processed_df['text'].apply(preprocess_for_tfidf)

    print(f"✅ Preprocessing complete!")
    return processed_df


def preprocess_text(text):
    """
    Main preprocessing function optimized for authorship attribution.

    Key decisions based on competition research:
    - Keep stopwords (they carry stylistic signals)
    - Light preprocessing to preserve author voice
    - Normalize while maintaining distinctiveness
    """
    if pd.isna(text):
        return ""

    # Convert to string and basic cleaning
    text = str(text)

    # Remove extra whitespace but preserve sentence structure
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Normalize quotes (handle Unicode quote characters safely)
    # Replace curly quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Lowercase for consistency (preserves most stylistic features)
    text = text.lower()

    return text


def preprocess_for_tfidf(text):
    """
    Preprocessing specifically optimized for TF-IDF features.
    More aggressive cleaning while preserving author signals.
    """
    text = preprocess_text(text)

    if SPACY_AVAILABLE:
        return preprocess_with_spacy(text)
    elif NLTK_AVAILABLE:
        return preprocess_with_nltk(text)
    else:
        return preprocess_basic(text)


def preprocess_with_spacy(text):
    """Advanced preprocessing using spaCy."""
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip punctuation and whitespace, but keep meaningful content
        if token.is_space or (token.is_punct and token.text not in ["'", '"']):
            continue

        # Lemmatize but keep stopwords (important for authorship)
        if token.lemma_ == "-PRON-":
            tokens.append(token.text)
        else:
            tokens.append(token.lemma_)

    return ' '.join(tokens)


def preprocess_with_nltk(text):
    """Preprocessing using NLTK when spaCy unavailable."""
    try:
        # Tokenize
        tokens = word_tokenize(text)

        # Remove pure punctuation but keep words with apostrophes
        tokens = [token for token in tokens
                  if not (token in string.punctuation and len(token) == 1)]

        # Lemmatize tokens (but keep stopwords!)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    except Exception as e:
        print(f"   ⚠️  NLTK preprocessing failed: {e}")
        print("   🔄 Falling back to basic preprocessing...")
        return preprocess_basic(text)


def preprocess_basic(text):
    """Basic preprocessing when neither spaCy nor NLTK available."""
    # Simple tokenization and cleaning
    text = re.sub(r'[^\w\s\']', ' ', text)  # Keep apostrophes
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text):
    """
    Tokenize text into list of tokens for neural models.
    Returns list of strings.
    """
    if pd.isna(text):
        return []

    text = preprocess_text(text)

    if SPACY_AVAILABLE:
        doc = nlp(text)
        return [token.text for token in doc if not token.is_space]
    elif NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception as e:
            print(f"   ⚠️  NLTK tokenization failed: {e}")
            print("   🔄 Falling back to basic tokenization...")
            return basic_tokenize(text)
    else:
        # Basic tokenization
        return basic_tokenize(text)


def basic_tokenize(text):
    """Basic tokenization fallback when NLTK/spaCy unavailable."""
    import re
    # Simple word tokenization using regex
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def extract_stylometric_features(text):
    """
    Extract author-specific stylometric features.
    These capture writing style beyond just word content.
    """
    if pd.isna(text):
        return {}

    text = str(text)

    # Basic text statistics
    char_count = len(text)
    word_count = len(text.split())
    sent_count = len(re.findall(r'[.!?]+', text))

    # Character-level features
    features = {
        'char_count': char_count,
        'word_count': word_count,
        'sent_count': max(1, sent_count),  # Avoid division by zero
        'avg_word_len': char_count / max(1, word_count),
        'avg_sent_len': word_count / max(1, sent_count),
    }

    # Punctuation features (important for author attribution)
    punct_features = extract_punctuation_features(text)
    features.update(punct_features)

    # Readability features
    readability_features = extract_readability_features(text, word_count, sent_count)
    features.update(readability_features)

    # POS features (if available)
    if SPACY_AVAILABLE or NLTK_AVAILABLE:
        pos_features = extract_pos_features(text)
        features.update(pos_features)

    return features


def extract_punctuation_features(text):
    """Extract punctuation-based features for author identification."""
    features = {}

    # Count different punctuation marks
    punct_counts = {
        'comma_count': text.count(','),
        'semicolon_count': text.count(';'),
        'colon_count': text.count(':'),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'quote_count': text.count('"') + text.count("'"),
        'dash_count': text.count('—') + text.count('--'),
        'paren_count': text.count('(') + text.count(')'),
        'period_count': text.count('.'),
    }

    # Normalize by text length to get ratios
    text_len = max(1, len(text))
    for feature_name, count in punct_counts.items():
        features[feature_name] = count
        features[f'{feature_name}_ratio'] = count / text_len

    return features


def extract_readability_features(text, word_count, sent_count):
    """Simple readability metrics for author classification."""
    features = {}

    # Syllable counting (approximation)
    syllable_count = estimate_syllables(text)

    # Flesch Reading Ease (approximation)
    if word_count > 0 and sent_count > 0:
        avg_sent_len = word_count / sent_count
        avg_syllables_per_word = syllable_count / word_count

        features['avg_syllables_per_word'] = avg_syllables_per_word
        features['flesch_score'] = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables_per_word)
    else:
        features['avg_syllables_per_word'] = 0
        features['flesch_score'] = 0

    return features


def estimate_syllables(text):
    """Simple syllable estimation for readability calculation."""
    # Count vowel groups as syllable approximation
    text = text.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    prev_char_vowel = False

    for char in text:
        if char in vowels:
            if not prev_char_vowel:
                syllable_count += 1
            prev_char_vowel = True
        else:
            prev_char_vowel = False

    # Adjust for silent e
    if text.endswith('e'):
        syllable_count -= 1

    return max(1, syllable_count)


def extract_pos_features(text):
    """Extract part-of-speech features for authorship attribution."""
    features = {}

    try:
        if SPACY_AVAILABLE:
            doc = nlp(text)
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

            total_tokens = len([t for t in doc if not t.is_space])
            if total_tokens > 0:
                for pos, count in pos_counts.items():
                    features[f'pos_{pos.lower()}_ratio'] = count / total_tokens

        elif NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)

                # Try different POS tagging approaches
                try:
                    pos_tags = nltk.pos_tag(tokens)
                except Exception as pos_error:
                    print(f"   ⚠️  Primary POS tagging failed: {pos_error}")
                    try:
                        # Try alternative POS tagging
                        pos_tags = nltk.pos_tag(tokens, tagset='universal')
                    except Exception as alt_pos_error:
                        print(f"   ⚠️  Alternative POS tagging failed: {alt_pos_error}")
                        # Return basic features without POS tags
                        return get_basic_linguistic_features(text)

                pos_counts = {}
                for _, pos in pos_tags:
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1

                total_tokens = len(pos_tags)
                if total_tokens > 0:
                    # Group similar POS tags
                    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'NOUN']
                    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB']
                    adj_tags = ['JJ', 'JJR', 'JJS', 'ADJ']
                    adv_tags = ['RB', 'RBR', 'RBS', 'ADV']

                    features['pos_noun_ratio'] = sum(pos_counts.get(tag, 0) for tag in noun_tags) / total_tokens
                    features['pos_verb_ratio'] = sum(pos_counts.get(tag, 0) for tag in verb_tags) / total_tokens
                    features['pos_adj_ratio'] = sum(pos_counts.get(tag, 0) for tag in adj_tags) / total_tokens
                    features['pos_adv_ratio'] = sum(pos_counts.get(tag, 0) for tag in adv_tags) / total_tokens

            except Exception as nltk_error:
                print(f"   ⚠️  NLTK POS feature extraction failed: {nltk_error}")
                return get_basic_linguistic_features(text)

    except Exception as e:
        print(f"   ⚠️  POS feature extraction failed: {e}")
        return get_basic_linguistic_features(text)

    return features


def get_basic_linguistic_features(text):
    """Get basic linguistic features when POS tagging fails."""
    features = {}

    words = text.lower().split()
    if len(words) > 0:
        # Basic word type approximations
        # Common nouns (very rough approximation)
        common_nouns = ['time', 'person', 'place', 'thing', 'way', 'day', 'man', 'world', 'life', 'hand']
        noun_count = sum(1 for word in words if word in common_nouns)
        features['pos_noun_ratio'] = noun_count / len(words)

        # Common verbs (rough approximation)
        common_verbs = ['is', 'was', 'are', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                        'should']
        verb_count = sum(1 for word in words if word in common_verbs)
        features['pos_verb_ratio'] = verb_count / len(words)

        # Simple approximations for adjectives and adverbs
        adj_endings = ['ed', 'ing', 'ful', 'less', 'ous', 'ive']
        adj_count = sum(1 for word in words if any(word.endswith(end) for end in adj_endings))
        features['pos_adj_ratio'] = adj_count / len(words)

        adv_endings = ['ly']
        adv_count = sum(1 for word in words if any(word.endswith(end) for end in adv_endings))
        features['pos_adv_ratio'] = adv_count / len(words)
    else:
        features['pos_noun_ratio'] = 0
        features['pos_verb_ratio'] = 0
        features['pos_adj_ratio'] = 0
        features['pos_adv_ratio'] = 0

    return features


def create_stylometric_dataframe(texts):
    """
    Create DataFrame of stylometric features for all texts.

    Args:
        texts: List or Series of text strings

    Returns:
        DataFrame with stylometric features
    """
    print(f"🎭 Extracting stylometric features for {len(texts):,} texts...")

    feature_dicts = [extract_stylometric_features(text) for text in texts]
    features_df = pd.DataFrame(feature_dicts)

    # Fill any missing values with 0
    features_df = features_df.fillna(0)

    print(f"✅ Extracted {features_df.shape[1]} stylometric features")
    return features_df