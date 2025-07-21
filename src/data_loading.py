# --- data_loading.py ---
"""
Data loading utilities for the Identify the Author competition.
Handles loading train.csv, test.csv, and sample_submission.csv with proper validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir: str):
    """
    Load competition datasets with validation and basic EDA.

    Args:
        data_dir: Directory containing train.csv, test.csv, sample_submission.csv

    Returns:
        tuple: (train_df, test_df, sample_sub_df)
    """
    data_path = Path(data_dir)

    # Load datasets
    print("Loading competition datasets...")

    # Try different possible locations for the files
    train_paths = [
        data_path / "train.csv",
        data_path / "train" / "train.csv"
    ]

    test_paths = [
        data_path / "test.csv",
        data_path / "test" / "test.csv"
    ]

    sample_paths = [
        data_path / "sample_submission.csv",
        data_path / "sample_submission" / "sample_submission.csv"
    ]

    # Find and load train data
    train_df = None
    for path in train_paths:
        if path.exists():
            train_df = pd.read_csv(path)
            print(f"✓ Loaded training data from {path}")
            break

    if train_df is None:
        # Create dummy training data for testing
        print("⚠️  No train.csv found. Creating dummy data for testing...")
        train_df = create_dummy_train_data()

    # Find and load test data
    test_df = None
    for path in test_paths:
        if path.exists():
            test_df = pd.read_csv(path)
            print(f"✓ Loaded test data from {path}")
            break

    if test_df is None:
        # Create dummy test data
        print("⚠️  No test.csv found. Creating dummy data for testing...")
        test_df = create_dummy_test_data()

    # Find and load sample submission
    sample_sub_df = None
    for path in sample_paths:
        if path.exists():
            sample_sub_df = pd.read_csv(path)
            print(f"✓ Loaded sample submission from {path}")
            break

    if sample_sub_df is None:
        print("⚠️  No sample_submission.csv found. Creating template...")
        sample_sub_df = create_sample_submission_template(test_df)

    # Validate and display basic info
    validate_and_display_data(train_df, test_df, sample_sub_df)

    return train_df, test_df, sample_sub_df


def create_dummy_train_data():
    """Create dummy training data for testing purposes."""
    # Create sample texts for each author
    sample_texts = {
        'EAP': [
            "Once upon a midnight dreary, while I pondered weak and weary.",
            "The boundaries which divide Life from Death are at best shadowy and vague.",
            "All that we see or seem is but a dream within a dream.",
            "Deep into that darkness peering, long I stood there wondering, fearing.",
            "And the raven, never flitting, still is sitting, still is sitting."
        ],
        'HPL': [
            "That is not dead which can eternal lie, and with strange aeons even death may die.",
            "The oldest and strongest emotion of mankind is fear, and the oldest fear is fear of the unknown.",
            "In his house at R'lyeh dead Cthulhu waits dreaming.",
            "The most merciful thing in the world is the inability of the human mind to correlate all its contents.",
            "Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn."
        ],
        'MWS': [
            "Nothing is so painful to the human mind as a great and sudden change.",
            "Life, although it may only be an accumulation of anguish, is dear to me.",
            "I do know that for the sympathy of one living being, I would make peace with all.",
            "The world was to me a secret which I desired to divine.",
            "Beware; for I am fearless, and therefore powerful."
        ]
    }

    # Generate 1000 samples with balanced classes
    n_samples = 1000
    samples_per_author = n_samples // 3

    data = []
    for i in range(n_samples):
        sample_id = f'id{i:05d}'

        # Determine author (roughly balanced)
        if i < samples_per_author:
            author = 'EAP'
        elif i < 2 * samples_per_author:
            author = 'HPL'
        else:
            author = 'MWS'

        # Select text (cycle through samples for variety)
        text_idx = i % len(sample_texts[author])
        text = sample_texts[author][text_idx]

        data.append({
            'id': sample_id,
            'text': text,
            'author': author
        })

    return pd.DataFrame(data)


def create_dummy_test_data():
    """Create dummy test data."""
    test_texts = [
        "The test sentence awaits classification by its mysterious author.",
        "Strange and eldritch forces shape the words upon this page.",
        "In solitude I find both comfort and the deepest melancholy.",
        "The ancient tome revealed secrets that mortal minds should never know.",
        "Upon reflection, the circumstances seemed most peculiar indeed."
    ]

    n_test_samples = 100
    data = []

    for i in range(n_test_samples):
        sample_id = f'id{i + 1000:05d}'  # Start from 1000 to avoid overlap
        text_idx = i % len(test_texts)
        text = test_texts[text_idx]

        data.append({
            'id': sample_id,
            'text': text
        })

    return pd.DataFrame(data)


def create_sample_submission_template(test_df):
    """Create sample submission template."""
    return pd.DataFrame({
        'id': test_df['id'].values,
        'EAP': [0.333333] * len(test_df),
        'HPL': [0.333333] * len(test_df),
        'MWS': [0.333334] * len(test_df)  # Ensure sum = 1
    })


def validate_and_display_data(train_df, test_df, sample_sub_df):
    """Validate datasets and display competition-specific insights."""

    print("\n" + "=" * 60)
    print("📊 COMPETITION DATA ANALYSIS")
    print("=" * 60)

    # Training data analysis
    print(f"\n🏋️ Training Data:")
    print(f"   Samples: {len(train_df):,}")
    print(f"   Features: {list(train_df.columns)}")

    if 'author' in train_df.columns:
        print(f"\n📝 Author Distribution:")
        author_counts = train_df['author'].value_counts()
        for author, count in author_counts.items():
            pct = count / len(train_df) * 100
            print(f"   {author}: {count:,} samples ({pct:.1f}%)")

        # Check class balance for CV strategy
        min_class = author_counts.min()
        max_class = author_counts.max()
        imbalance_ratio = max_class / min_class
        if imbalance_ratio > 1.5:
            print(f"   ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
            print(f"   💡 Consider stratified sampling in cross-validation")

    # Text length analysis (critical for single-sentence data)
    if 'text' in train_df.columns:
        text_lengths = train_df['text'].str.len()
        word_counts = train_df['text'].str.split().str.len()

        print(f"\n📏 Text Characteristics:")
        print(f"   Character length: {text_lengths.mean():.0f} ± {text_lengths.std():.0f}")
        print(f"   Word count: {word_counts.mean():.1f} ± {word_counts.std():.1f}")
        print(f"   Range: {text_lengths.min()}-{text_lengths.max()} characters")

        # Short text warning (important for feature engineering)
        if text_lengths.mean() < 100:
            print(f"   📝 Note: Short texts detected - optimize for sentence-level features")

    # Test data analysis
    print(f"\n🧪 Test Data:")
    print(f"   Samples: {len(test_df):,}")
    print(f"   Features: {list(test_df.columns)}")

    # Submission format validation
    print(f"\n📤 Submission Format:")
    print(f"   Expected IDs: {len(sample_sub_df):,}")
    print(f"   Probability columns: {[col for col in sample_sub_df.columns if col != 'id']}")

    # Data integrity checks
    print(f"\n✅ Data Integrity:")

    # Check for missing values
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    print(f"   Missing values - Train: {train_missing}, Test: {test_missing}")

    # Check ID overlap (should be none)
    if 'id' in train_df.columns and 'id' in test_df.columns:
        id_overlap = set(train_df['id']) & set(test_df['id'])
        if id_overlap:
            print(f"   ⚠️  ID overlap detected: {len(id_overlap)} samples")
        else:
            print(f"   ✓ No train/test ID overlap")

    # Verify test IDs match submission template
    if 'id' in test_df.columns and 'id' in sample_sub_df.columns:
        missing_ids = set(test_df['id']) - set(sample_sub_df['id'])
        extra_ids = set(sample_sub_df['id']) - set(test_df['id'])

        if missing_ids or extra_ids:
            print(f"   ⚠️  Submission ID mismatch - Missing: {len(missing_ids)}, Extra: {len(extra_ids)}")
        else:
            print(f"   ✓ Test/submission IDs match perfectly")

    print(f"\n🎯 Ready for preprocessing and feature engineering!")
    print("=" * 60)


def load_competition_train_data():
    """
    Convenience function to load only training data.
    Useful for notebooks and quick experimentation.
    """
    train_df, _, _ = load_data('./data')
    return train_df