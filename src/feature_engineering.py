# --- feature_engineering.py ---
"""
Feature engineering module optimized for authorship attribution.
Combines TF-IDF, character n-grams, stylometric features, and embeddings.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from text_preprocessing import create_stylometric_dataframe


def build_features(train_df, test_df):
    """
    Build comprehensive feature set for authorship attribution.

    Args:
        train_df: Training DataFrame with preprocessed text
        test_df: Test DataFrame with preprocessed text

    Returns:
        tuple: (X_train, X_test, feature_info)
    """
    print("🔧 Building multi-modal feature set for authorship attribution...")

    # Initialize feature info dictionary
    feature_info = {
        'feature_types': [],
        'feature_counts': {},
        'scalers': {},
        'vectorizers': {},
        'selectors': {}
    }

    # Combine datasets for consistent feature extraction
    all_texts_tfidf = pd.concat([train_df['text_for_tfidf'], test_df['text_for_tfidf']], axis=0)
    all_texts_original = pd.concat([train_df['text'], test_df['text']], axis=0)

    n_train = len(train_df)

    feature_matrices = []

    # 1. TF-IDF Features (Word N-grams)
    print("   📝 Extracting TF-IDF word features...")
    word_features, word_vectorizer = extract_word_tfidf_features(
        all_texts_tfidf, n_train
    )
    feature_matrices.append(word_features)
    feature_info['vectorizers']['word_tfidf'] = word_vectorizer
    feature_info['feature_counts']['word_tfidf'] = word_features['train'].shape[1]
    feature_info['feature_types'].append('word_tfidf')

    # 2. Character N-gram Features
    print("   🔤 Extracting character n-gram features...")
    char_features, char_vectorizer = extract_char_tfidf_features(
        all_texts_original, n_train
    )
    feature_matrices.append(char_features)
    feature_info['vectorizers']['char_tfidf'] = char_vectorizer
    feature_info['feature_counts']['char_tfidf'] = char_features['train'].shape[1]
    feature_info['feature_types'].append('char_tfidf')

    # 3. Stylometric Features
    print("   🎭 Extracting stylometric features...")
    style_features, style_scaler = extract_stylometric_features(
        all_texts_original, n_train
    )
    feature_matrices.append(style_features)
    feature_info['scalers']['stylometric'] = style_scaler
    feature_info['feature_counts']['stylometric'] = style_features['train'].shape[1]
    feature_info['feature_types'].append('stylometric')

    # 4. Dense Embeddings (if possible)
    print("   🧠 Extracting embedding features...")
    try:
        embedding_features, embedding_scaler = extract_embedding_features(
            all_texts_tfidf, n_train
        )
        feature_matrices.append(embedding_features)
        feature_info['scalers']['embeddings'] = embedding_scaler
        feature_info['feature_counts']['embeddings'] = embedding_features['train'].shape[1]
        feature_info['feature_types'].append('embeddings')
    except Exception as e:
        print(f"   ⚠️  Skipping embeddings: {e}")

    # Combine all features
    print("   🔗 Combining feature matrices...")
    X_train, X_test = combine_feature_matrices(feature_matrices)

    # Feature selection (optional, for dimensionality reduction)
    if 'author' in train_df.columns:
        print("   ✂️  Applying feature selection...")
        X_train, X_test, selector = apply_feature_selection(
            X_train, X_test, train_df['author'], max_features=50000
        )
        feature_info['selectors']['main'] = selector

    print(f"✅ Feature engineering complete!")
    print(f"   Final feature matrix: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")

    return X_train, X_test, feature_info


def extract_word_tfidf_features(texts, n_train):
    """Extract TF-IDF features from word n-grams."""

    # Optimized for authorship attribution
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=20000,  # Reasonable limit for memory
        min_df=3,  # Remove very rare words
        max_df=0.95,  # Remove very common words
        sublinear_tf=True,  # Apply log scaling
        use_idf=True,  # Use inverse document frequency
        smooth_idf=True,  # Smooth IDF to avoid zero division
        norm='l2',  # L2 normalization
        lowercase=True,  # Ensure lowercase (should already be done)
        stop_words=None,  # Keep stopwords (important for authorship!)
        token_pattern=r'\b\w+\b'  # Standard word tokens
    )

    # Fit on all data to ensure consistent vocabulary
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Split back to train/test
    train_features = tfidf_matrix[:n_train]
    test_features = tfidf_matrix[n_train:]

    return {
        'train': train_features,
        'test': test_features
    }, vectorizer


def extract_char_tfidf_features(texts, n_train):
    """Extract TF-IDF features from character n-grams."""

    # Character-level features are excellent for authorship attribution
    vectorizer = TfidfVectorizer(
        analyzer='char',  # Character-level analysis
        ngram_range=(3, 5),  # 3-5 character n-grams (optimal range)
        max_features=30000,  # More features for character level
        min_df=2,  # Keep slightly rarer character patterns
        max_df=0.98,  # Remove very common patterns
        sublinear_tf=True,  # Log scaling
        use_idf=True,
        smooth_idf=True,
        norm='l2',
        lowercase=True
    )

    # Fit on all data
    char_tfidf_matrix = vectorizer.fit_transform(texts)

    # Split back to train/test
    train_features = char_tfidf_matrix[:n_train]
    test_features = char_tfidf_matrix[n_train:]

    return {
        'train': train_features,
        'test': test_features
    }, vectorizer


def extract_stylometric_features(texts, n_train):
    """Extract and scale stylometric features."""

    # Create stylometric feature DataFrame
    style_df = create_stylometric_dataframe(texts)

    # Split train/test
    train_style = style_df.iloc[:n_train]
    test_style = style_df.iloc[n_train:]

    # Scale features for neural models and distance-based algorithms
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_style)
    test_scaled = scaler.transform(test_style)

    return {
        'train': train_scaled,
        'test': test_scaled
    }, scaler


def extract_embedding_features(texts, n_train):
    """
    Extract dense embedding features.
    Uses simple word averaging as baseline (can be enhanced with BERT later).
    """
    try:
        # Try to use pre-trained embeddings if available
        embeddings = extract_glove_embeddings(texts)
    except:
        # Fallback to TF-IDF + SVD for dense features
        embeddings = extract_tfidf_svd_embeddings(texts)

    # Split train/test
    train_embeddings = embeddings[:n_train]
    test_embeddings = embeddings[n_train:]

    # Scale embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    return {
        'train': train_scaled,
        'test': test_scaled
    }, scaler


def extract_glove_embeddings(texts):
    """
    Extract GloVe embeddings if available.
    Falls back to random embeddings for demonstration.
    """
    # This is a placeholder - in production you'd load actual GloVe vectors
    print("   📦 Using averaged word embeddings (placeholder)")

    # Create dummy embeddings for demonstration
    n_samples = len(texts)
    embedding_dim = 100

    # Generate consistent embeddings based on text hash
    embeddings = []
    for text in texts:
        # Simple hash-based embedding generation for demo
        text_hash = hash(str(text)) % 1000000
        np.random.seed(text_hash % 10000)  # Consistent seed based on text
        embedding = np.random.normal(0, 0.1, embedding_dim)
        embeddings.append(embedding)

    return np.array(embeddings)


def extract_tfidf_svd_embeddings(texts):
    """Create dense embeddings using TF-IDF + SVD."""
    print("   📊 Creating TF-IDF + SVD embeddings")

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=None  # Keep stopwords for authorship
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    return embeddings


def combine_feature_matrices(feature_matrices):
    """
    Combine different types of feature matrices.
    Handles both sparse and dense matrices.
    """
    train_matrices = []
    test_matrices = []

    for feature_dict in feature_matrices:
        train_matrices.append(feature_dict['train'])
        test_matrices.append(feature_dict['test'])

    # Combine sparse and dense matrices
    X_train = combine_sparse_dense_matrices(train_matrices)
    X_test = combine_sparse_dense_matrices(test_matrices)

    return X_train, X_test


def combine_sparse_dense_matrices(matrices):
    """
    Efficiently combine sparse and dense matrices.
    """
    if len(matrices) == 1:
        return matrices[0]

    # Convert all to same format (prefer sparse if any are sparse)
    has_sparse = any(sparse.issparse(m) for m in matrices)

    if has_sparse:
        # Convert dense to sparse and concatenate
        sparse_matrices = []
        for m in matrices:
            if sparse.issparse(m):
                sparse_matrices.append(m)
            else:
                sparse_matrices.append(sparse.csr_matrix(m))

        return sparse.hstack(sparse_matrices)
    else:
        # All dense matrices
        return np.hstack(matrices)


def apply_feature_selection(X_train, X_test, y_train, max_features=50000):
    """
    Apply feature selection to reduce dimensionality.
    Uses chi-squared test for feature selection.
    """
    if X_train.shape[1] <= max_features:
        print(f"   ✓ No feature selection needed ({X_train.shape[1]} <= {max_features})")
        return X_train, X_test, None

    print(f"   ✂️  Selecting top {max_features} features from {X_train.shape[1]}...")

    # Use chi-squared test for feature selection
    selector = SelectKBest(chi2, k=min(max_features, X_train.shape[1]))

    # Ensure non-negative values for chi2 test
    if sparse.issparse(X_train):
        X_train_pos = X_train.copy()
        X_train_pos.data = np.abs(X_train_pos.data)
        X_test_pos = X_test.copy()
        X_test_pos.data = np.abs(X_test_pos.data)
    else:
        X_train_pos = np.abs(X_train)
        X_test_pos = np.abs(X_test)

    # Fit selector and transform
    X_train_selected = selector.fit_transform(X_train_pos, y_train)
    X_test_selected = selector.transform(X_test_pos)

    print(f"   ✅ Selected {X_train_selected.shape[1]} features")

    return X_train_selected, X_test_selected, selector


def create_bert_features(texts, model_name='bert-base-uncased', max_length=128):
    """
    Create BERT embeddings for texts.
    This is a placeholder - actual implementation would use transformers library.
    """
    print(f"   🤖 Creating BERT features (placeholder)")

    # Placeholder for BERT embeddings
    n_samples = len(texts)
    bert_dim = 768

    # Create dummy BERT-like embeddings
    embeddings = []
    for i, text in enumerate(texts):
        # Use text hash for consistent embeddings
        text_hash = hash(str(text)) % 1000000
        np.random.seed(text_hash % 10000)
        embedding = np.random.normal(0, 0.1, bert_dim)
        embeddings.append(embedding)

    return np.array(embeddings)


def extract_advanced_stylometric_features(texts):
    """
    Extract advanced stylometric features for authorship attribution.
    """
    features_list = []

    for text in texts:
        features = {}

        # Lexical richness measures
        words = text.lower().split()
        unique_words = set(words)

        if len(words) > 0:
            features['type_token_ratio'] = len(unique_words) / len(words)
            features['hapax_legomena_ratio'] = sum(1 for word in unique_words if words.count(word) == 1) / len(words)
        else:
            features['type_token_ratio'] = 0
            features['hapax_legomena_ratio'] = 0

        # Function word ratios (important for authorship)
        function_words = ['the', 'of', 'and', 'to', 'in', 'a', 'is', 'it', 'for', 'as']
        if len(words) > 0:
            features['function_word_ratio'] = sum(1 for word in words if word in function_words) / len(words)
        else:
            features['function_word_ratio'] = 0

        # Syntactic complexity
        features['avg_words_per_sentence'] = len(words) / max(1, text.count('.') + text.count('!') + text.count('?'))

        features_list.append(features)

    return pd.DataFrame(features_list)


def get_feature_importance_summary(feature_info, model=None):
    """
    Generate a summary of feature types and their importance.
    """
    summary = {
        'total_features': sum(feature_info['feature_counts'].values()),
        'feature_breakdown': feature_info['feature_counts'].copy(),
        'feature_types': feature_info['feature_types'].copy()
    }

    print("\n📊 Feature Engineering Summary:")
    print("=" * 50)

    for feature_type, count in summary['feature_breakdown'].items():
        pct = (count / summary['total_features']) * 100
        print(f"{feature_type:15}: {count:6,} features ({pct:5.1f}%)")

    print("-" * 50)
    print(f"{'Total':15}: {summary['total_features']:6,} features (100.0%)")

    return summary