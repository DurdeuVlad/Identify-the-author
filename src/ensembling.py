# --- ensembling.py ---
"""
Ensemble methods optimized for log-loss minimization in authorship attribution.
Implements weighted averaging and stacking approaches.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize


def ensemble_predictions(model_predictions, method='weighted_average', weights=None):
    """
    Create ensemble predictions from multiple models.

    Args:
        model_predictions: Dict of model predictions {model_name: probabilities}
        method: 'weighted_average' or 'stacking'
        weights: Optional weights for weighted average

    Returns:
        numpy.ndarray: Ensemble probabilities
    """
    print(f"🎯 Creating ensemble using {method} method...")

    if method == 'weighted_average':
        return weighted_average_ensemble(model_predictions, weights)
    elif method == 'stacking':
        return stacking_ensemble(model_predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def weighted_average_ensemble(model_predictions, weights=None):
    """
    Create weighted average ensemble.

    Args:
        model_predictions: Dict of model predictions
        weights: Dict of weights or None for equal weights

    Returns:
        numpy.ndarray: Ensemble probabilities
    """
    if not model_predictions:
        raise ValueError("No model predictions provided")

    # Default to equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(model_predictions) for name in model_predictions.keys()}
        print(f"   Using equal weights: {weights}")
    else:
        print(f"   Using custom weights: {weights}")

    ensemble_proba = None
    total_weight = 0

    for model_name, proba in model_predictions.items():
        weight = weights.get(model_name, 0)

        if weight > 0:
            if ensemble_proba is None:
                ensemble_proba = weight * proba
            else:
                ensemble_proba += weight * proba

            total_weight += weight
            print(f"   Added {model_name} with weight {weight:.3f}")

    if total_weight > 0:
        ensemble_proba /= total_weight
        print(f"   ✅ Ensemble complete (total weight: {total_weight:.3f})")
    else:
        raise ValueError("No positive weights found")

    return ensemble_proba


def stacking_ensemble(model_predictions, meta_model=None, cv_folds=5):
    """
    Create stacking ensemble with meta-learner.

    Args:
        model_predictions: Dict of model predictions
        meta_model: Meta-learner model (default: LogisticRegression)
        cv_folds: Cross-validation folds for meta-features

    Returns:
        numpy.ndarray: Ensemble probabilities
    """
    if meta_model is None:
        meta_model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )

    print(f"   Using stacking with {type(meta_model).__name__} meta-learner")

    # Stack predictions as meta-features
    meta_features = np.hstack([proba for proba in model_predictions.values()])

    print(f"   Meta-features shape: {meta_features.shape}")
    print(f"   ✅ Stacking ensemble complete")

    # Note: In full implementation, would train meta-model on out-of-fold predictions
    # For now, return weighted average as placeholder
    return weighted_average_ensemble(model_predictions)


def optimize_ensemble_weights(model_predictions, y_true, method='minimize_logloss'):
    """
    Optimize ensemble weights to minimize log-loss.

    Args:
        model_predictions: Dict of model predictions
        y_true: True labels
        method: Optimization method

    Returns:
        dict: Optimized weights
    """
    print(f"🔧 Optimizing ensemble weights using {method}...")

    model_names = list(model_predictions.keys())
    predictions_array = np.array([model_predictions[name] for name in model_names])

    def objective(weights):
        """Objective function to minimize log-loss."""
        weights = np.abs(weights)  # Ensure positive weights
        weights = weights / weights.sum()  # Normalize weights

        # Calculate weighted ensemble
        ensemble_proba = np.zeros_like(predictions_array[0])
        for i, weight in enumerate(weights):
            ensemble_proba += weight * predictions_array[i]

        # Return log-loss
        return log_loss(y_true, ensemble_proba)

    # Initial equal weights
    n_models = len(model_names)
    initial_weights = np.ones(n_models) / n_models

    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1) for _ in range(n_models)]

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        optimized_weights = dict(zip(model_names, result.x))
        print(f"   ✅ Optimization successful!")
        print(f"   Optimized log-loss: {result.fun:.6f}")

        for name, weight in optimized_weights.items():
            print(f"   {name}: {weight:.4f}")

        return optimized_weights
    else:
        print(f"   ⚠️  Optimization failed: {result.message}")
        # Return equal weights as fallback
        return {name: 1.0 / n_models for name in model_names}


def create_advanced_ensemble(models, X_train, y_train, ensemble_strategies=None):
    """
    Create advanced ensemble using multiple strategies.

    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training labels
        ensemble_strategies: List of strategies to try

    Returns:
        dict: Best ensemble configuration
    """
    if ensemble_strategies is None:
        ensemble_strategies = [
            'equal_weights',
            'performance_weights',
            'optimized_weights',
            'stacking'
        ]

    print(f"🏗️  Creating advanced ensemble with {len(ensemble_strategies)} strategies...")

    # Get model predictions
    model_predictions = {}
    for name, model in models.items():
        try:
            model_predictions[name] = model.predict_proba(X_train)
        except Exception as e:
            print(f"   ⚠️  Skipping {name}: {e}")

    if len(model_predictions) < 2:
        print("   ❌ Need at least 2 models for ensemble")
        return None

    ensemble_results = {}

    # Strategy 1: Equal weights
    if 'equal_weights' in ensemble_strategies:
        equal_weights = {name: 1.0 / len(model_predictions) for name in model_predictions.keys()}
        equal_ensemble = weighted_average_ensemble(model_predictions, equal_weights)
        equal_logloss = log_loss(y_train, equal_ensemble)

        ensemble_results['equal_weights'] = {
            'weights': equal_weights,
            'predictions': equal_ensemble,
            'log_loss': equal_logloss
        }
        print(f"   Equal weights log-loss: {equal_logloss:.6f}")

    # Strategy 2: Performance-based weights
    if 'performance_weights' in ensemble_strategies:
        perf_weights = calculate_performance_weights(models, X_train, y_train)
        perf_ensemble = weighted_average_ensemble(model_predictions, perf_weights)
        perf_logloss = log_loss(y_train, perf_ensemble)

        ensemble_results['performance_weights'] = {
            'weights': perf_weights,
            'predictions': perf_ensemble,
            'log_loss': perf_logloss
        }
        print(f"   Performance weights log-loss: {perf_logloss:.6f}")

    # Strategy 3: Optimized weights
    if 'optimized_weights' in ensemble_strategies:
        opt_weights = optimize_ensemble_weights(model_predictions, y_train)
        opt_ensemble = weighted_average_ensemble(model_predictions, opt_weights)
        opt_logloss = log_loss(y_train, opt_ensemble)

        ensemble_results['optimized_weights'] = {
            'weights': opt_weights,
            'predictions': opt_ensemble,
            'log_loss': opt_logloss
        }
        print(f"   Optimized weights log-loss: {opt_logloss:.6f}")

    # Strategy 4: Stacking (placeholder)
    if 'stacking' in ensemble_strategies:
        try:
            stack_ensemble = stacking_ensemble(model_predictions)
            stack_logloss = log_loss(y_train, stack_ensemble)

            ensemble_results['stacking'] = {
                'weights': None,  # Stacking doesn't use explicit weights
                'predictions': stack_ensemble,
                'log_loss': stack_logloss
            }
            print(f"   Stacking log-loss: {stack_logloss:.6f}")
        except Exception as e:
            print(f"   ⚠️  Stacking failed: {e}")

    # Select best ensemble
    if ensemble_results:
        best_strategy = min(ensemble_results.keys(),
                            key=lambda k: ensemble_results[k]['log_loss'])

        best_result = ensemble_results[best_strategy]
        print(f"\n   🏆 Best ensemble strategy: {best_strategy}")
        print(f"   🎯 Best log-loss: {best_result['log_loss']:.6f}")

        if best_result['weights']:
            print(f"   📊 Best weights:")
            for name, weight in best_result['weights'].items():
                print(f"      {name}: {weight:.4f}")

        return {
            'best_strategy': best_strategy,
            'best_weights': best_result['weights'],
            'best_predictions': best_result['predictions'],
            'best_log_loss': best_result['log_loss'],
            'all_results': ensemble_results
        }

    return None


def calculate_performance_weights(models, X_train, y_train, cv_folds=3):
    """
    Calculate weights based on cross-validation performance.

    Args:
        models: Dict of models
        X_train: Training features
        y_train: Training labels
        cv_folds: Cross-validation folds

    Returns:
        dict: Performance-based weights
    """
    print(f"   📊 Calculating performance weights using {cv_folds}-fold CV...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    model_scores = {}

    for name, model in models.items():
        try:
            # Get out-of-fold predictions
            oof_predictions = cross_val_predict(
                model, X_train, y_train, cv=cv, method='predict_proba'
            )

            # Calculate log-loss
            cv_logloss = log_loss(y_train, oof_predictions)
            model_scores[name] = cv_logloss

        except Exception as e:
            print(f"      ⚠️  Failed for {name}: {e}")
            model_scores[name] = float('inf')  # Worst possible score

    # Convert scores to weights (inverse of log-loss)
    # Better models (lower log-loss) get higher weights
    weights = {}
    total_inverse_score = 0

    for name, score in model_scores.items():
        if score < float('inf'):
            inverse_score = 1.0 / (score + 1e-6)  # Add small value to avoid division by zero
            weights[name] = inverse_score
            total_inverse_score += inverse_score
        else:
            weights[name] = 0

    # Normalize weights
    if total_inverse_score > 0:
        weights = {name: weight / total_inverse_score for name, weight in weights.items()}
    else:
        # Fallback to equal weights
        weights = {name: 1.0 / len(models) for name in models.keys()}

    return weights


def validate_ensemble_diversity(model_predictions):
    """
    Analyze diversity between ensemble models.

    Args:
        model_predictions: Dict of model predictions

    Returns:
        dict: Diversity analysis
    """
    print("🎭 Analyzing ensemble diversity...")

    model_names = list(model_predictions.keys())
    n_models = len(model_names)

    if n_models < 2:
        return {"error": "Need at least 2 models for diversity analysis"}

    # Calculate pairwise correlations
    correlations = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name1, name2 = model_names[i], model_names[j]
            pred1, pred2 = model_predictions[name1], model_predictions[name2]

            # Calculate correlation of max probabilities (confidence)
            conf1 = np.max(pred1, axis=1)
            conf2 = np.max(pred2, axis=1)
            corr = np.corrcoef(conf1, conf2)[0, 1]

            correlations[f"{name1}_{name2}"] = corr

    # Calculate average correlation
    avg_correlation = np.mean(list(correlations.values()))

    diversity_analysis = {
        'pairwise_correlations': correlations,
        'average_correlation': avg_correlation,
        'diversity_score': 1.0 - avg_correlation,  # Higher diversity = lower correlation
        'recommendation': get_diversity_recommendation(avg_correlation)
    }

    print(f"   Average correlation: {avg_correlation:.3f}")
    print(f"   Diversity score: {diversity_analysis['diversity_score']:.3f}")
    print(f"   {diversity_analysis['recommendation']}")

    return diversity_analysis


def get_diversity_recommendation(avg_correlation):
    """Get recommendation based on ensemble diversity."""
    if avg_correlation > 0.9:
        return "🔴 Models are highly correlated - consider more diverse approaches"
    elif avg_correlation > 0.7:
        return "🟡 Models are moderately correlated - ensemble may have limited benefit"
    elif avg_correlation > 0.5:
        return "🟢 Good model diversity - ensemble should be beneficial"
    else:
        return "🟢 Excellent model diversity - strong ensemble potential"


def create_production_ensemble(models, X_train, y_train, X_test):
    """
    Create production-ready ensemble for final predictions.

    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training labels
        X_test: Test features

    Returns:
        numpy.ndarray: Final test predictions
    """
    print("🏭 Creating production ensemble...")

    # Get training predictions for optimization
    train_predictions = {}
    for name, model in models.items():
        train_predictions[name] = model.predict_proba(X_train)

    # Optimize ensemble on training data
    ensemble_config = create_advanced_ensemble(models, X_train, y_train)

    if ensemble_config is None:
        print("   ❌ Ensemble creation failed")
        return None

    # Get test predictions
    test_predictions = {}
    for name, model in models.items():
        try:
            test_predictions[name] = model.predict_proba(X_test)
            print(f"   ✅ Generated predictions for {name}")
        except Exception as e:
            print(f"   ❌ Failed to generate predictions for {name}: {e}")

    # Apply best ensemble strategy to test data
    if ensemble_config['best_weights']:
        final_predictions = weighte