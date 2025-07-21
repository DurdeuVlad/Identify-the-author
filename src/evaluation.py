# --- evaluation.py ---
"""
Evaluation and calibration module for authorship attribution models.
Focuses on log-loss optimization and probability calibration.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt


def evaluate_models(models, X_train, y_train, cv_folds=5):
    """
    Comprehensive evaluation of all models using cross-validation.

    Args:
        models: Dictionary of trained models
        X_train: Training feature matrix
        y_train: Training labels
        cv_folds: Number of cross-validation folds
    """
    print(f"📊 Evaluating {len(models)} models with {cv_folds}-fold cross-validation...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    print("\n" + "=" * 80)
    print(f"{'Model':<12} {'Log-Loss':<12} {'Std':<8} {'Accuracy':<12} {'Std':<8} {'Status'}")
    print("-" * 80)

    for name, model in models.items():
        try:
            # Evaluate log-loss (primary metric)
            log_loss_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='neg_log_loss', n_jobs=-1
            )
            log_loss_mean = -log_loss_scores.mean()
            log_loss_std = log_loss_scores.std()

            # Evaluate accuracy (secondary metric)
            accuracy_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1
            )
            accuracy_mean = accuracy_scores.mean()
            accuracy_std = accuracy_scores.std()

            results[name] = {
                'log_loss_mean': log_loss_mean,
                'log_loss_std': log_loss_std,
                'accuracy_mean': accuracy_mean,
                'accuracy_std': accuracy_std,
                'log_loss_scores': -log_loss_scores,
                'accuracy_scores': accuracy_scores
            }

            status = "✅ OK"
            print(f"{name:<12} {log_loss_mean:<12.4f} {log_loss_std:<8.4f} "
                  f"{accuracy_mean:<12.3f} {accuracy_std:<8.3f} {status}")

        except Exception as e:
            results[name] = {'error': str(e)}
            status = f"❌ ERROR: {str(e)[:20]}..."
            print(f"{name:<12} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'N/A':<8} {status}")

    print("=" * 80)

    # Rank models by log-loss
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model = min(valid_results.keys(), key=lambda k: valid_results[k]['log_loss_mean'])
        print(f"\n🏆 Best model by log-loss: {best_model} "
              f"({valid_results[best_model]['log_loss_mean']:.4f})")

    return results


def calibrate_model(model, X_train, y_train, method='isotonic', cv_folds=3):
    """
    Calibrate model probabilities for better log-loss performance.

    Args:
        model: Trained model to calibrate
        X_train: Training features
        y_train: Training labels
        method: Calibration method ('isotonic' or 'sigmoid')
        cv_folds: Cross-validation folds for calibration

    Returns:
        Calibrated model
    """
    print(f"🎯 Calibrating model probabilities using {method} method...")

    # Check if model needs calibration
    if needs_calibration(model, X_train, y_train):
        calibrated = CalibratedClassifierCV(
            model,
            method=method,
            cv=cv_folds,
            ensemble=False  # Use single calibrated model
        )
        calibrated.fit(X_train, y_train)
        print(f"   ✅ Model calibrated")
        return calibrated
    else:
        print(f"   ✓ Model already well-calibrated")
        return model


def needs_calibration(model, X_train, y_train, threshold=0.02):
    """
    Check if model needs calibration by examining reliability.

    Args:
        model: Model to check
        X_train: Training features
        y_train: Training labels
        threshold: Reliability threshold

    Returns:
        bool: True if calibration needed
    """
    try:
        # Use a small holdout set to check calibration
        from sklearn.model_selection import train_test_split

        X_cal, X_test, y_cal, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        model.fit(X_cal, y_cal)
        y_proba = model.predict_proba(X_test)

        # Check calibration for each class
        total_reliability_error = 0
        n_classes = len(np.unique(y_train))

        for class_idx in range(n_classes):
            y_true_binary = (y_test == model.classes_[class_idx]).astype(int)
            y_prob_class = y_proba[:, class_idx]

            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_prob_class, n_bins=5, normalize=False
                )

                # Calculate reliability error
                reliability_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                total_reliability_error += reliability_error

            except Exception:
                # If calibration curve fails, assume calibration needed
                return True

        avg_reliability_error = total_reliability_error / n_classes
        return avg_reliability_error > threshold

    except Exception:
        # If anything fails, err on side of calibration
        return True


def analyze_model_performance(model, X_train, y_train, model_name="Model"):
    """
    Detailed performance analysis for a single model.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        model_name: Name for reporting
    """
    print(f"\n🔍 Detailed Analysis: {model_name}")
    print("=" * 50)

    # Get predictions
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)

    # Basic metrics
    accuracy = accuracy_score(y_train, y_pred)
    logloss = log_loss(y_train, y_proba)

    print(f"Training Accuracy: {accuracy:.3f}")
    print(f"Training Log-Loss: {logloss:.4f}")

    # Per-class analysis
    print(f"\nPer-Class Performance:")
    print(classification_report(y_train, y_pred, target_names=model.classes_))

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_train, y_pred)
    print(cm)

    # Class probability distributions
    print(f"\nProbability Distribution Analysis:")
    for i, class_name in enumerate(model.classes_):
        class_probs = y_proba[:, i]
        print(f"{class_name}: mean={class_probs.mean():.3f}, "
              f"std={class_probs.std():.3f}, "
              f"min={class_probs.min():.3f}, "
              f"max={class_probs.max():.3f}")


def plot_calibration_analysis(models, X_train, y_train):
    """
    Plot calibration curves for all models.

    Args:
        models: Dictionary of models
        X_train: Training features
        y_train: Training labels
    """
    print("📈 Generating calibration plots...")

    try:
        from sklearn.model_selection import train_test_split

        # Use holdout set for calibration analysis
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        n_models = len(models)
        n_classes = len(np.unique(y_train))

        fig, axes = plt.subplots(n_classes, n_models, figsize=(4 * n_models, 4 * n_classes))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        if n_classes == 1:
            axes = axes.reshape(1, -1)

        class_names = np.unique(y_train)

        for model_idx, (model_name, model) in enumerate(models.items()):
            try:
                model.fit(X_cal, y_cal)
                y_proba = model.predict_proba(X_test)

                for class_idx, class_name in enumerate(class_names):
                    ax = axes[class_idx, model_idx]

                    y_true_binary = (y_test == class_name).astype(int)
                    y_prob_class = y_proba[:, class_idx]

                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true_binary, y_prob_class, n_bins=10
                    )

                    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
                    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    ax.set_xlabel("Mean Predicted Probability")
                    ax.set_ylabel("Fraction of Positives")
                    ax.set_title(f"{model_name} - {class_name}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"   ⚠️  Calibration plot failed for {model_name}: {e}")

        plt.tight_layout()
        plt.savefig('calibration_plots.png', dpi=150, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"   ⚠️  Calibration plotting failed: {e}")


def cross_validate_ensemble(models, X_train, y_train, ensemble_weights=None):
    """
    Cross-validate ensemble performance.

    Args:
        models: Dictionary of models
        X_train: Training features
        y_train: Training labels
        ensemble_weights: Optional weights for models

    Returns:
        dict: Ensemble cross-validation results
    """
    print("🎯 Cross-validating ensemble performance...")

    if ensemble_weights is None:
        ensemble_weights = {name: 1.0 / len(models) for name in models.keys()}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Train all models on fold
        fold_predictions = {}
        for name, model in models.items():
            try:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                fold_predictions[name] = model_copy.predict_proba(X_fold_val)
            except Exception as e:
                print(f"   ⚠️  Model {name} failed in fold {fold_idx}: {e}")
                continue

        if len(fold_predictions) > 0:
            # Ensemble predictions
            ensemble_proba = ensemble_predictions_weighted(fold_predictions, ensemble_weights)
            fold_logloss = log_loss(y_fold_val, ensemble_proba)
            fold_scores.append(fold_logloss)

    if fold_scores:
        mean_logloss = np.mean(fold_scores)
        std_logloss = np.std(fold_scores)

        print(f"   Ensemble CV Log-Loss: {mean_logloss:.4f} ± {std_logloss:.4f}")

        return {
            'mean_log_loss': mean_logloss,
            'std_log_loss': std_logloss,
            'fold_scores': fold_scores
        }
    else:
        print("   ❌ Ensemble cross-validation failed")
        return None


def ensemble_predictions_weighted(predictions_dict, weights):
    """
    Create weighted ensemble predictions.

    Args:
        predictions_dict: Dictionary of model predictions
        weights: Dictionary of model weights

    Returns:
        numpy.ndarray: Ensemble probabilities
    """
    ensemble_proba = None
    total_weight = 0

    for model_name, proba in predictions_dict.items():
        weight = weights.get(model_name, 0)
        if weight > 0:
            if ensemble_proba is None:
                ensemble_proba = weight * proba
            else:
                ensemble_proba += weight * proba
            total_weight += weight

    if total_weight > 0:
        ensemble_proba /= total_weight

    return ensemble_proba


def evaluate_prediction_confidence(models, X_train, y_train):
    """
    Analyze prediction confidence patterns across models.

    Args:
        models: Dictionary of models
        X_train: Training features
        y_train: Training labels
    """
    print("🎯 Analyzing prediction confidence patterns...")

    confidence_analysis = {}

    for model_name, model in models.items():
        try:
            y_proba = model.predict_proba(X_train)

            # Calculate confidence metrics
            max_proba = np.max(y_proba, axis=1)  # Confidence = highest probability
            entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)  # Uncertainty

            confidence_analysis[model_name] = {
                'mean_confidence': np.mean(max_proba),
                'std_confidence': np.std(max_proba),
                'mean_entropy': np.mean(entropy),
                'std_entropy': np.std(entropy),
                'min_confidence': np.min(max_proba),
                'max_confidence': np.max(max_proba)
            }

        except Exception as e:
            print(f"   ⚠️  Confidence analysis failed for {model_name}: {e}")

    # Display results
    print("\nConfidence Analysis Results:")
    print("-" * 70)
    print(f"{'Model':<12} {'Mean Conf':<10} {'Std Conf':<10} {'Mean Ent':<10} {'Range'}")
    print("-" * 70)

    for model_name, stats in confidence_analysis.items():
        conf_range = f"{stats['min_confidence']:.2f}-{stats['max_confidence']:.2f}"
        print(f"{model_name:<12} {stats['mean_confidence']:<10.3f} "
              f"{stats['std_confidence']:<10.3f} {stats['mean_entropy']:<10.3f} {conf_range}")

    return confidence_analysis


def generate_evaluation_report(models, evaluation_results, X_train, y_train):
    """
    Generate comprehensive evaluation report.

    Args:
        models: Dictionary of models
        evaluation_results: Results from evaluate_models
        X_train: Training features
        y_train: Training labels

    Returns:
        dict: Comprehensive evaluation report
    """
    print("📋 Generating comprehensive evaluation report...")

    report = {
        'dataset_info': {
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'class_distribution': pd.Series(y_train).value_counts().to_dict()
        },
        'model_performance': evaluation_results,
        'recommendations': []
    }

    # Add recommendations based on results
    valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}

    if valid_results:
        # Best single model
        best_model = min(valid_results.keys(),
                         key=lambda k: valid_results[k]['log_loss_mean'])

        report['best_single_model'] = best_model
        report['best_log_loss'] = valid_results[best_model]['log_loss_mean']

        # Model selection recommendations
        top_models = sorted(valid_results.keys(),
                            key=lambda k: valid_results[k]['log_loss_mean'])[:3]

        report['top_models'] = top_models
        report['recommendations'].append(f"Top 3 models for ensembling: {', '.join(top_models)}")

        # Calibration recommendations
        for model_name in top_models:
            if needs_calibration(models[model_name], X_train, y_train):
                report['recommendations'].append(f"Consider calibrating {model_name}")

    # Feature engineering recommendations
    if X_train.shape[1] > 50000:
        report['recommendations'].append("Consider feature selection for high-dimensional data")

    if X_train.shape[0] < 1000:
        report['recommendations'].append("Small dataset - focus on regularization and cross-validation")

    print("✅ Evaluation report generated!")
    return report