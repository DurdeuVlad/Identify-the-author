# --- submission.py ---
"""
Submission generation module for Kaggle competition.
Handles probability clipping, formatting, and validation.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os


def make_submission(ids, probas, sample_submission_df, output_path='submission.csv'):
    """
    Create Kaggle-formatted submission file.

    Args:
        ids: Test sample IDs
        probas: Predicted probabilities (n_samples x n_classes)
        sample_submission_df: Sample submission format
        output_path: Output file path

    Returns:
        pandas.DataFrame: Formatted submission
    """
    print("📤 Creating Kaggle submission file...")

    # Validate inputs
    print(f"   Test samples: {len(ids):,}")
    print(f"   Probability matrix shape: {probas.shape}")

    if len(ids) != probas.shape[0]:
        raise ValueError(f"ID count ({len(ids)}) doesn't match probability count ({probas.shape[0]})")

    # Get class names from sample submission
    class_columns = [col for col in sample_submission_df.columns if col != 'id']
    expected_classes = len(class_columns)

    if probas.shape[1] != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, got {probas.shape[1]}")

    print(f"   Classes: {class_columns}")

    # Apply probability clipping for numerical stability
    clipped_probas = clip_probabilities(probas)

    # Ensure probabilities sum to 1 (important for log-loss)
    normalized_probas = normalize_probabilities(clipped_probas)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': ids
    })

    # Add probability columns
    for i, class_name in enumerate(class_columns):
        submission_df[class_name] = normalized_probas[:, i]

    # Validate submission format
    validate_submission(submission_df, sample_submission_df)

    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"   ✅ Submission saved to: {output_path}")

    # Generate summary
    generate_submission_summary(submission_df, class_columns, output_path)

    return submission_df


def clip_probabilities(probas, min_prob=1e-15, max_prob=1 - 1e-15):
    """
    Clip probabilities to avoid log-loss explosion.

    Args:
        probas: Probability matrix
        min_prob: Minimum probability value
        max_prob: Maximum probability value

    Returns:
        numpy.ndarray: Clipped probabilities
    """
    print(f"   🔧 Clipping probabilities to [{min_prob:.2e}, {max_prob:.2e}]")

    original_min = probas.min()
    original_max = probas.max()

    clipped = np.clip(probas, min_prob, max_prob)

    clipped_min = clipped.min()
    clipped_max = clipped.max()

    print(f"      Original range: [{original_min:.6f}, {original_max:.6f}]")
    print(f"      Clipped range:  [{clipped_min:.6f}, {clipped_max:.6f}]")

    return clipped


def normalize_probabilities(probas):
    """
    Ensure probabilities sum to 1 for each sample.

    Args:
        probas: Probability matrix

    Returns:
        numpy.ndarray: Normalized probabilities
    """
    print("   🎯 Normalizing probabilities to sum to 1...")

    # Calculate row sums
    row_sums = probas.sum(axis=1, keepdims=True)

    # Check for zero sums (shouldn't happen with clipping)
    zero_sums = (row_sums == 0).sum()
    if zero_sums > 0:
        print(f"      ⚠️  Warning: {zero_sums} samples have zero probability sum")
        row_sums = np.where(row_sums == 0, 1, row_sums)

    # Normalize
    normalized = probas / row_sums

    # Verify normalization
    final_sums = normalized.sum(axis=1)
    min_sum = final_sums.min()
    max_sum = final_sums.max()

    print(f"      Row sum range: [{min_sum:.10f}, {max_sum:.10f}]")

    if abs(min_sum - 1.0) > 1e-10 or abs(max_sum - 1.0) > 1e-10:
        print(f"      ⚠️  Warning: Normalization not perfect")
    else:
        print(f"      ✅ Probabilities properly normalized")

    return normalized


def validate_submission(submission_df, sample_submission_df):
    """
    Validate submission format against sample submission.

    Args:
        submission_df: Generated submission
        sample_submission_df: Sample submission format
    """
    print("   ✅ Validating submission format...")

    # Check columns
    expected_cols = set(sample_submission_df.columns)
    actual_cols = set(submission_df.columns)

    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        error_msg = ""
        if missing:
            error_msg += f"Missing columns: {missing}. "
        if extra:
            error_msg += f"Extra columns: {extra}. "

        raise ValueError(f"Column mismatch. {error_msg}")

    # Check row count
    if len(submission_df) != len(sample_submission_df):
        raise ValueError(f"Row count mismatch: expected {len(sample_submission_df)}, got {len(submission_df)}")

    # Check ID overlap
    expected_ids = set(sample_submission_df['id'])
    actual_ids = set(submission_df['id'])

    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids

    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} IDs in submission")
    if extra_ids:
        raise ValueError(f"Found {len(extra_ids)} unexpected IDs in submission")

    # Check probability values
    prob_columns = [col for col in submission_df.columns if col != 'id']
    prob_values = submission_df[prob_columns].values

    # Check for NaN or infinite values
    if np.isnan(prob_values).any():
        raise ValueError("Found NaN values in probabilities")
    if np.isinf(prob_values).any():
        raise ValueError("Found infinite values in probabilities")

    # Check probability ranges
    if (prob_values < 0).any():
        raise ValueError("Found negative probabilities")
    if (prob_values > 1).any():
        raise ValueError("Found probabilities > 1")

    # Check row sums
    row_sums = prob_values.sum(axis=1)
    tolerance = 1e-10

    bad_sums = np.abs(row_sums - 1.0) > tolerance
    if bad_sums.any():
        n_bad = bad_sums.sum()
        print(f"      ⚠️  Warning: {n_bad} rows don't sum to 1 (within {tolerance} tolerance)")

    print("      ✅ Submission format validated")


def generate_submission_summary(submission_df, class_columns, output_path):
    """
    Generate summary statistics for the submission.

    Args:
        submission_df: Submission DataFrame
        class_columns: List of class column names
        output_path: Submission file path
    """
    print("\n   📊 Submission Summary:")
    print("   " + "=" * 50)

    # Basic info
    print(f"   File: {output_path}")
    print(f"   Samples: {len(submission_df):,}")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Probability statistics per class
    print(f"\n   Class Probability Statistics:")
    for class_name in class_columns:
        probs = submission_df[class_name].values
        print(f"   {class_name}: mean={probs.mean():.4f}, "
              f"std={probs.std():.4f}, "
              f"min={probs.min():.6f}, "
              f"max={probs.max():.6f}")

    # Confidence distribution
    max_probs = submission_df[class_columns].max(axis=1)
    print(f"\n   Prediction Confidence:")
    print(f"   Mean max probability: {max_probs.mean():.4f}")
    print(f"   Std max probability:  {max_probs.std():.4f}")
    print(f"   Min confidence:       {max_probs.min():.4f}")
    print(f"   Max confidence:       {max_probs.max():.4f}")

    # Predicted class distribution
    predicted_classes = submission_df[class_columns].idxmax(axis=1)
    class_counts = predicted_classes.value_counts()

    print(f"\n   Predicted Class Distribution:")
    for class_name, count in class_counts.items():
        pct = count / len(submission_df) * 100
        print(f"   {class_name}: {count:,} samples ({pct:.1f}%)")

    # File size
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    print(f"\n   File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    print("   " + "=" * 50)


def create_ensemble_submission(model_predictions, test_ids, sample_submission_df,
                               ensemble_weights=None, output_path='ensemble_submission.csv'):
    """
    Create submission from ensemble of model predictions.

    Args:
        model_predictions: Dict of {model_name: predictions}
        test_ids: Test sample IDs
        sample_submission_df: Sample submission format
        ensemble_weights: Optional weights for models
        output_path: Output file path

    Returns:
        pandas.DataFrame: Ensemble submission
    """
    print(f"🎯 Creating ensemble submission from {len(model_predictions)} models...")

    # Ensemble predictions
    if ensemble_weights is None:
        ensemble_weights = {name: 1.0 / len(model_predictions) for name in model_predictions.keys()}

    ensemble_proba = None
    total_weight = 0

    for model_name, proba in model_predictions.items():
        weight = ensemble_weights.get(model_name, 0)
        if weight > 0:
            if ensemble_proba is None:
                ensemble_proba = weight * proba
            else:
                ensemble_proba += weight * proba
            total_weight += weight
            print(f"   {model_name}: weight={weight:.3f}")

    if total_weight > 0:
        ensemble_proba /= total_weight

    # Create submission
    return make_submission(test_ids, ensemble_proba, sample_submission_df, output_path)


def generate_multiple_submissions(models, X_test, test_ids, sample_submission_df, output_dir='submissions'):
    """
    Generate individual and ensemble submissions.

    Args:
        models: Dict of trained models
        X_test: Test features
        test_ids: Test sample IDs
        sample_submission_df: Sample submission format
        output_dir: Output directory

    Returns:
        dict: Generated submission files
    """
    print(f"📦 Generating multiple submissions in {output_dir}/...")

    os.makedirs(output_dir, exist_ok=True)

    submission_files = {}
    model_predictions = {}

    # Generate individual model submissions
    for model_name, model in models.items():
        try:
            predictions = model.predict_proba(X_test)
            model_predictions[model_name] = predictions

            output_path = os.path.join(output_dir, f'submission_{model_name}.csv')
            submission_df = make_submission(test_ids, predictions, sample_submission_df, output_path)
            submission_files[model_name] = output_path

            print(f"   ✅ Created {model_name} submission")

        except Exception as e:
            print(f"   ❌ Failed to create {model_name} submission: {e}")

    # Generate ensemble submissions
    if len(model_predictions) > 1:
        # Equal weight ensemble
        ensemble_path = os.path.join(output_dir, 'submission_ensemble_equal.csv')
        ensemble_submission = create_ensemble_submission(
            model_predictions, test_ids, sample_submission_df,
            ensemble_weights=None, output_path=ensemble_path
        )
        submission_files['ensemble_equal'] = ensemble_path

        print(f"   ✅ Created equal-weight ensemble submission")

    print(f"\n   📋 Generated {len(submission_files)} submission files:")
    for name, path in submission_files.items():
        print(f"      {name}: {path}")

    return submission_files


def validate_competition_submission(submission_path, expected_format):
    """
    Final validation of submission file for competition.

    Args:
        submission_path: Path to submission file
        expected_format: Dict with expected format info

    Returns:
        bool: True if valid
    """
    print(f"🔍 Final validation of {submission_path}...")

    try:
        # Load submission
        df = pd.read_csv(submission_path)

        # Check basic format
        if 'id' not in df.columns:
            print("   ❌ Missing 'id' column")
            return False

        # Check expected columns
        expected_cols = expected_format.get('columns', ['id', 'EAP', 'HPL', 'MWS'])
        if list(df.columns) != expected_cols:
            print(f"   ❌ Column mismatch. Expected: {expected_cols}, Got: {list(df.columns)}")
            return False

        # Check row count
        expected_rows = expected_format.get('n_rows')
        if expected_rows and len(df) != expected_rows:
            print(f"   ❌ Row count mismatch. Expected: {expected_rows}, Got: {len(df)}")
            return False

        # Check for duplicated IDs
        if df['id'].duplicated().any():
            print("   ❌ Found duplicated IDs")
            return False

        # Check probability values
        prob_cols = [col for col in df.columns if col != 'id']
        prob_values = df[prob_cols].values

        if np.isnan(prob_values).any() or np.isinf(prob_values).any():
            print("   ❌ Found NaN or infinite values")
            return False

        if (prob_values < 0).any() or (prob_values > 1).any():
            print("   ❌ Probabilities out of [0,1] range")
            return False

        # Check row sums
        row_sums = prob_values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            print("   ❌ Probabilities don't sum to 1")
            return False

        print("   ✅ Submission validation passed!")
        return True

    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False