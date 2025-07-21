# --- model_training.py ---
"""
Model training module for authorship attribution.
Implements multiple complementary models optimized for log-loss.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier

    SKLEARN_MLP_AVAILABLE = True
except ImportError:
    SKLEARN_MLP_AVAILABLE = False


def train_models(X_train, y_train, cv_folds=5):
    """
    Train multiple models for authorship attribution ensemble.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        cv_folds: Number of cross-validation folds

    Returns:
        tuple: (models_dict, predictions_dict)
    """
    print(f"🏋️ Training models for authorship attribution...")
    print(f"   Training data: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
    print(f"   Cross-validation: {cv_folds}-fold stratified")

    # Show unique labels for reference
    unique_labels = sorted(set(y_train))
    print(f"   Authors: {unique_labels}")

    # Initialize models dictionary
    models = {}
    predictions = {}

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # 1. Logistic Regression (Strong baseline for text classification)
    print("\n   📊 Training Logistic Regression...")
    lr_model = create_logistic_regression_model(X_train)
    models['lr'] = lr_model
    lr_scores = evaluate_model_cv(lr_model, X_train, y_train, cv)
    print(f"      CV Log-loss: {lr_scores['log_loss']:.4f} ± {lr_scores['log_loss_std']:.4f}")

    # 2. Naive Bayes (Fast baseline, good with sparse features)
    print("\n   🚀 Training Naive Bayes...")
    nb_model = create_naive_bayes_model(X_train)
    models['nb'] = nb_model
    nb_scores = evaluate_model_cv(nb_model, X_train, y_train, cv)
    print(f"      CV Log-loss: {nb_scores['log_loss']:.4f} ± {nb_scores['log_loss_std']:.4f}")

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n   🌳 Training XGBoost...")
        xgb_model = create_xgboost_model(X_train)
        models['xgb'] = xgb_model
        # XGBoost wrapper handles label encoding internally
        xgb_scores = evaluate_model_cv(xgb_model, X_train, y_train, cv)
        print(f"      CV Log-loss: {xgb_scores['log_loss']:.4f} ± {xgb_scores['log_loss_std']:.4f}")

    # 4. Support Vector Machine
    if X_train.shape[1] < 10000:  # Only for smaller feature sets
        print("\n   ⚖️  Training SVM...")
        svm_model = create_svm_model(X_train)
        models['svm'] = svm_model
        svm_scores = evaluate_model_cv(svm_model, X_train, y_train, cv)
        print(f"      CV Log-loss: {svm_scores['log_loss']:.4f} ± {svm_scores['log_loss_std']:.4f}")

    # 5. Random Forest (for feature importance analysis)
    print("\n   🌲 Training Random Forest...")
    rf_model = create_random_forest_model(X_train)
    models['rf'] = rf_model
    rf_scores = evaluate_model_cv(rf_model, X_train, y_train, cv)
    print(f"      CV Log-loss: {rf_scores['log_loss']:.4f} ± {rf_scores['log_loss_std']:.4f}")

    # 6. Neural Network (if features are manageable)
    if SKLEARN_MLP_AVAILABLE and X_train.shape[1] < 5000:
        print("\n   🧠 Training Neural Network...")
        nn_model = create_neural_network_model(X_train)
        models['nn'] = nn_model
        nn_scores = evaluate_model_cv(nn_model, X_train, y_train, cv)
        print(f"      CV Log-loss: {nn_scores['log_loss']:.4f} ± {nn_scores['log_loss_std']:.4f}")

    # Train final models on full data
    print("\n   🔧 Training final models on full dataset...")
    final_models = {}
    for name, model in models.items():
        print(f"      Training {name}...")

        # All models now handle string labels properly
        final_model = model.fit(X_train, y_train)
        final_models[name] = final_model

        # Store predictions for ensemble
        predictions[name] = final_model.predict_proba(X_train)

    print(f"\n✅ Model training complete! Trained {len(final_models)} models.")

    return final_models, predictions


def create_logistic_regression_model(X_train):
    """Create optimized Logistic Regression for authorship attribution."""

    # Determine if data is sparse
    is_sparse = hasattr(X_train, 'nnz')

    # Configure based on dataset size and type
    if X_train.shape[1] > 10000:
        # High-dimensional text data - use L1 regularization
        model = LogisticRegression(
            C=1.0,  # Regularization strength
            penalty='l1',  # L1 for feature selection
            solver='liblinear',  # Handles L1 penalty
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
            # Removed deprecated multi_class parameter
        )
    else:
        # Lower dimensional - use L2 regularization
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',  # Good for small datasets
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
            # Removed deprecated multi_class parameter
        )

    return model


def create_naive_bayes_model(X_train):
    """Create optimized Naive Bayes model."""

    # Use Complement Naive Bayes for imbalanced text data
    if hasattr(X_train, 'nnz'):  # Sparse matrix
        # For TF-IDF features, Complement NB often works better
        model = ComplementNB(alpha=1.0, norm=False)
    else:
        # For dense features, use standard Multinomial NB
        model = MultinomialNB(alpha=1.0)

    return model


class XGBoostWrapper:
    """Wrapper for XGBoost to handle string labels properly and be sklearn-compatible."""

    def __init__(self, **xgb_params):
        """Initialize with XGBoost parameters."""
        from sklearn.preprocessing import LabelEncoder

        # Store XGBoost parameters
        self.xgb_params = xgb_params
        self.label_encoder = LabelEncoder()
        self.xgb_model = None
        self.classes_ = None
        self._is_fitted = False

    def fit(self, X, y):
        """Fit using encoded labels."""
        # Fit label encoder if not already fitted
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)

        self.classes_ = self.label_encoder.classes_

        # Encode labels if they are strings
        if isinstance(y[0], str):
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y

        # Create fresh XGBoost model with parameters
        self.xgb_model = XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(X, y_encoded)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict and return original string labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        y_encoded = self.xgb_model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        """Predict probabilities (same order as original classes)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self.xgb_model.predict_proba(X)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return self.xgb_params.copy()

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.xgb_params.update(params)
        return self


def create_xgboost_model(X_train, label_encoder=None):
    """Create optimized XGBoost model for authorship attribution."""

    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")

    # Check for GPU availability
    try:
        import xgboost as xgb
        gpu_available = xgb.get_config()['use_gpu'] if hasattr(xgb, 'get_config') else False
    except:
        gpu_available = False

    # XGBoost parameters with GPU optimization
    xgb_params = {
        'objective': 'multi:softprob',  # Output probabilities (good for log-loss)
        'eval_metric': 'mlogloss',  # Optimize log-loss directly
        'n_estimators': 100,  # Conservative number of trees
        'max_depth': 6,  # Prevent overfitting
        'learning_rate': 0.1,
        'subsample': 0.8,  # Row sampling
        'colsample_bytree': 0.8,  # Feature sampling
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'verbosity': 0  # Quiet output
    }

    # Add GPU parameters if available
    if gpu_available:
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'n_jobs': 1  # GPU mode uses single job
        })
        print("      🚀 XGBoost GPU acceleration enabled!")
    else:
        xgb_params['n_jobs'] = -1  # Use all CPU cores
        print("      ⚡ XGBoost CPU mode (GPU not available)")

    # Return wrapper that handles string labels
    return XGBoostWrapper(**xgb_params)


def create_svm_model(X_train):
    """Create SVM model with probability estimates."""

    model = SVC(
        C=1.0,  # Regularization parameter
        kernel='linear',  # Linear kernel for text data
        probability=True,  # Enable probability estimates
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )

    return model


def create_random_forest_model(X_train):
    """Create Random Forest model optimized for text classification."""

    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=None,  # Grow trees fully
        min_samples_split=5,  # Minimum samples to split
        min_samples_leaf=2,  # Minimum samples in leaf
        max_features='sqrt',  # Feature sampling
        bootstrap=True,  # Bootstrap sampling
        oob_score=True,  # Out-of-bag scoring
        random_state=42,
        n_jobs=-1,  # Use all cores
        class_weight='balanced'  # Handle imbalanced classes
    )

    return model


def create_neural_network_model(X_train):
    """Create neural network model for dense features."""

    if not SKLEARN_MLP_AVAILABLE:
        raise ImportError("MLPClassifier not available")

    # Configure based on feature count
    if X_train.shape[1] < 100:
        hidden_layers = (50, 25)
    elif X_train.shape[1] < 1000:
        hidden_layers = (100, 50)
    else:
        hidden_layers = (200, 100, 50)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',  # ReLU activation
        solver='adam',  # Adam optimizer
        alpha=0.01,  # L2 regularization
        batch_size='auto',  # Automatic batch size
        learning_rate='adaptive',  # Adaptive learning rate
        learning_rate_init=0.001,
        max_iter=500,  # Maximum iterations
        shuffle=True,  # Shuffle data
        random_state=42,
        early_stopping=True,  # Early stopping
        validation_fraction=0.1,  # Validation set for early stopping
        n_iter_no_change=10,  # Patience for early stopping
        tol=1e-4  # Tolerance for optimization
    )

    return model


def evaluate_model_cv(model, X, y, cv):
    """
    Evaluate model using cross-validation.

    Returns:
        dict: Cross-validation scores
    """
    # Log-loss (primary metric)
    log_loss_scores = cross_val_score(
        model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1
    )
    log_loss_mean = -log_loss_scores.mean()
    log_loss_std = log_loss_scores.std()

    # Accuracy (for reference)
    accuracy_scores = cross_val_score(
        model, X, y, cv=cv, scoring='accuracy', n_jobs=-1
    )
    accuracy_mean = accuracy_scores.mean()
    accuracy_std = accuracy_scores.std()

    return {
        'log_loss': log_loss_mean,
        'log_loss_std': log_loss_std,
        'accuracy': accuracy_mean,
        'accuracy_std': accuracy_std,
        'raw_log_loss_scores': -log_loss_scores,
        'raw_accuracy_scores': accuracy_scores
    }


def create_bert_model():
    """
    Create BERT model for fine-tuning.
    This is a placeholder - actual implementation would use transformers library.
    """
    print("📖 BERT model creation - placeholder")

    # Placeholder for BERT implementation
    class BERTPlaceholder:
        def __init__(self):
            self.is_fitted = False

        def fit(self, X, y):
            print("   🤖 Fine-tuning BERT (placeholder)")
            self.classes_ = np.unique(y)
            self.is_fitted = True
            return self

        def predict_proba(self, X):
            if not self.is_fitted:
                raise ValueError("Model not fitted")

            # Return dummy probabilities
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            n_classes = len(self.classes_)

            # Generate consistent dummy probabilities
            np.random.seed(42)
            probs = np.random.dirichlet([1] * n_classes, n_samples)

            return probs

        def predict(self, X):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]

    return BERTPlaceholder()


def get_model_performance_summary(models, X_train, y_train):
    """
    Generate performance summary for all trained models.
    """
    print("\n📊 Model Performance Summary:")
    print("=" * 70)
    print(f"{'Model':<15} {'Log-Loss':<12} {'Accuracy':<12} {'Features':<10}")
    print("-" * 70)

    results = {}

    for name, model in models.items():
        # Get predictions
        try:
            y_pred_proba = model.predict_proba(X_train)
            y_pred = model.predict(X_train)

            # Calculate metrics
            train_log_loss = log_loss(y_train, y_pred_proba)
            train_accuracy = accuracy_score(y_train, y_pred)

            print(f"{name:<15} {train_log_loss:<12.4f} {train_accuracy:<12.3f} {X_train.shape[1]:<10,}")

            results[name] = {
                'log_loss': train_log_loss,
                'accuracy': train_accuracy,
                'n_features': X_train.shape[1]
            }

        except Exception as e:
            print(f"{name:<15} {'ERROR':<12} {'ERROR':<12} {X_train.shape[1]:<10,}")
            results[name] = {'error': str(e)}

    print("=" * 70)
    print("Note: These are training metrics. Use cross-validation for unbiased estimates.")

    return results


def select_best_models(models, cv_results, top_k=3):
    """
    Select the best performing models for ensembling.

    Args:
        models: Dictionary of trained models
        cv_results: Cross-validation results
        top_k: Number of top models to select

    Returns:
        dict: Selected models
    """
    # This would be implemented based on actual CV results
    # For now, return all models
    return models