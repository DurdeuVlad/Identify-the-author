# src/model_training.py - TensorFlow-Free Version
"""
Model training module for Identify the Author competition.
NO TENSORFLOW - Uses PyTorch + XGBoost + Scikit-learn only.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("📦 XGBoost not available - install with: pip install xgboost")

try:
    from sklearn.neural_network import MLPClassifier

    SKLEARN_MLP_AVAILABLE = True
except ImportError:
    SKLEARN_MLP_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("📦 Transformers not available - install with: pip install transformers torch")


def train_models(X_train, y_train, feature_type='tfidf'):
    """
    Train multiple models for author classification.

    Args:
        X_train: Training features
        y_train: Training labels (can be strings or integers)
        feature_type: Type of features ('tfidf', 'dense', 'mixed')

    Returns:
        dict: Dictionary of trained models with label encoder
    """
    print(f"🤖 Training models on {feature_type} features...")
    print(f"   Training set: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")

    # Handle string labels for XGBoost
    label_encoder = None
    y_encoded = y_train

    if isinstance(y_train[0], str):
        print(f"   🔤 Encoding string labels: {np.unique(y_train)}")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)
        print(f"   🔢 Encoded to: {np.unique(y_encoded)}")

    models = {}

    # Model 1: Logistic Regression (Handles string labels natively)
    print("\n1️⃣ Training Logistic Regression...")
    models['logistic'] = create_logistic_regression_model()
    models['logistic'].fit(X_train, y_train)  # Use original labels
    print("   ✅ Logistic Regression trained")

    # Model 2: Naive Bayes (Choose appropriate variant)
    if feature_type == 'counts':
        print("\n2️⃣ Training Multinomial Naive Bayes...")
        models['naive_bayes'] = create_naive_bayes_model(variant='multinomial')
    elif feature_type in ['tfidf', 'dense', 'mixed']:
        print("\n2️⃣ Training Gaussian Naive Bayes...")
        models['naive_bayes'] = create_naive_bayes_model(variant='gaussian')
        print("   📝 Note: Using Gaussian NB (handles negative TF-IDF values)")

    models['naive_bayes'].fit(X_train, y_train)  # Use original labels
    print("   ✅ Naive Bayes trained")

    # Model 3: XGBoost (Requires numeric labels)
    if XGBOOST_AVAILABLE:
        print("\n3️⃣ Training XGBoost...")
        xgb_wrapper = XGBoostWrapper()
        xgb_wrapper.fit(X_train, y_train)  # Wrapper handles label encoding
        models['xgboost'] = xgb_wrapper
        print("   ✅ XGBoost trained")

    # Model 4: Random Forest (Handles string labels natively)
    print("\n4️⃣ Training Random Forest...")
    models['random_forest'] = create_random_forest_model(X_train)
    models['random_forest'].fit(X_train, y_train)  # Use original labels
    print("   ✅ Random Forest trained")

    # Model 5: BERT (PyTorch-based, NO TensorFlow)
    if TRANSFORMERS_AVAILABLE and feature_type == 'text':
        print("\n5️⃣ Training BERT (PyTorch)...")
        models['bert'] = create_pytorch_bert_model()
        # Note: BERT training would be implemented separately
        print("   ⚡ BERT model created (fine-tuning requires separate implementation)")

    print(f"\n🎯 Trained {len(models)} models successfully!")
    return models


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for XGBoost to handle string labels."""

    def __init__(self, xgb_model=None, label_encoder=None, original_classes=None):
        self.xgb_model = xgb_model
        self.label_encoder = label_encoder
        self.original_classes = original_classes
        if original_classes is not None:
            self.classes_ = original_classes

    def fit(self, X, y):
        """Fit the model with proper label handling."""
        if self.xgb_model is None:
            self.xgb_model = create_xgboost_model()

        # Handle string labels
        if isinstance(y[0], str):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.original_classes = self.label_encoder.classes_
            self.classes_ = self.original_classes
        else:
            y_encoded = y
            self.classes_ = np.unique(y)

        self.xgb_model.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        """Get probabilities and return in original class order."""
        probs = self.xgb_model.predict_proba(X)
        return probs

    def predict(self, X):
        """Get predictions and convert back to original labels."""
        encoded_preds = self.xgb_model.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_preds.astype(int))
        return encoded_preds

    def get_params(self, deep=True):
        """Get parameters for scikit-learn compatibility."""
        return {
            'xgb_model': self.xgb_model,
            'label_encoder': self.label_encoder,
            'original_classes': self.original_classes
        }

    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_logistic_regression_model():
    """Create optimized logistic regression for author attribution."""
    model = LogisticRegression(
        C=1.0,  # Regularization strength
        penalty='l2',  # L2 regularization
        solver='liblinear',  # Good for small datasets
        multi_class='ovr',  # One-vs-rest for 3 classes
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )
    return model


def create_naive_bayes_model(variant='multinomial'):
    """Create Naive Bayes model - choose variant based on feature type."""
    if variant == 'multinomial':
        # For count-based features (non-negative)
        model = MultinomialNB(
            alpha=1.0,  # Laplace smoothing
            fit_prior=True,  # Learn class priors
            class_prior=None  # Use training data frequencies
        )
    elif variant == 'gaussian':
        # For continuous features (can handle negative values)
        base_model = GaussianNB(
            priors=None,  # Learn from data
            var_smoothing=1e-9  # Smoothing parameter
        )

        # Wrap in a class that handles sparse matrices
        class GaussianNBWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, var_smoothing=1e-9):
                self.var_smoothing = var_smoothing
                self.base_model = GaussianNB(
                    priors=None,
                    var_smoothing=var_smoothing
                )

            def fit(self, X, y):
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                self.base_model.fit(X_dense, y)
                self.classes_ = self.base_model.classes_
                return self

            def predict(self, X):
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                return self.base_model.predict(X_dense)

            def predict_proba(self, X):
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                return self.base_model.predict_proba(X_dense)

            def get_params(self, deep=True):
                """Get parameters for scikit-learn compatibility."""
                return {'var_smoothing': self.var_smoothing}

            def set_params(self, **params):
                """Set parameters for scikit-learn compatibility."""
                for key, value in params.items():
                    setattr(self, key, value)
                # Recreate base model with new parameters
                self.base_model = GaussianNB(
                    priors=None,
                    var_smoothing=self.var_smoothing
                )
                return self

        model = GaussianNBWrapper()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model


def create_xgboost_model():
    """Create XGBoost model for any feature type."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")

    model = xgb.XGBClassifier(
        objective='multi:softprob',  # Multi-class probabilities
        n_estimators=100,  # Number of trees
        max_depth=6,  # Tree depth
        learning_rate=0.1,  # Learning rate
        subsample=0.8,  # Row sampling
        colsample_bytree=0.8,  # Feature sampling
        random_state=42,
        n_jobs=-1,  # Use all cores
        eval_metric='mlogloss',  # Multi-class log-loss
        verbosity=0  # Suppress XGBoost warnings
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


def create_pytorch_bert_model():
    """
    Create BERT model using PyTorch (NO TensorFlow).
    Returns a wrapper that can be fine-tuned separately.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("   ⚠️ Transformers not available - returning placeholder")
        return create_bert_placeholder()

    class PyTorchBERTWrapper:
        """Wrapper for BERT using PyTorch backend."""

        def __init__(self):
            self.model_name = 'bert-base-uncased'
            self.tokenizer = None
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_fitted = False

        def initialize(self):
            """Initialize BERT model and tokenizer."""
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3  # EAP, HPL, MWS
                )
                self.model.to(self.device)
                print(f"   ✅ BERT initialized on {self.device}")
                return True
            except Exception as e:
                print(f"   ❌ BERT initialization failed: {e}")
                return False

        def fit(self, X, y):
            """Placeholder for BERT fine-tuning."""
            print("   🤖 BERT fine-tuning (requires separate training loop)")
            # This would implement the full training loop
            self.classes_ = np.unique(y)
            self.is_fitted = True
            return self

        def predict_proba(self, X):
            """Generate predictions using BERT."""
            if not self.is_fitted:
                raise ValueError("Model not fitted")

            # For now, return dummy probabilities
            # Real implementation would tokenize texts and run inference
            n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
            np.random.seed(42)
            probs = np.random.dirichlet([1, 1, 1], n_samples)
            return probs

        def predict(self, X):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]

    return PyTorchBERTWrapper()


def create_bert_placeholder():
    """Fallback placeholder when transformers not available."""

    class BERTPlaceholder:
        def __init__(self):
            self.is_fitted = False

        def fit(self, X, y):
            print("   🤖 BERT placeholder (install transformers for real BERT)")
            self.classes_ = np.unique(y)
            self.is_fitted = True
            return self

        def predict_proba(self, X):
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            np.random.seed(42)
            return np.random.dirichlet([1, 1, 1], n_samples)

        def predict(self, X):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]

    return BERTPlaceholder()


def evaluate_model_cv(model, X, y, cv=5):
    """
    Evaluate model using cross-validation with log-loss focus.

    Returns:
        dict: Cross-validation scores
    """
    # Create stratified CV
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Log-loss (primary metric for competition)
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


def get_model_performance_summary(models, X_train, y_train):
    """Generate performance summary for all trained models."""
    print("\n📊 Model Performance Summary:")
    print("=" * 70)
    print(f"{'Model':<20} {'Log-Loss':<12} {'Accuracy':<12} {'Features':<10}")
    print("-" * 70)

    results = {}

    for name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_train)
            y_pred = model.predict(X_train)

            train_log_loss = log_loss(y_train, y_pred_proba)
            train_accuracy = accuracy_score(y_train, y_pred)

            print(f"{name:<20} {train_log_loss:<12.4f} {train_accuracy:<12.3f} {X_train.shape[1]:<10,}")

            results[name] = {
                'log_loss': train_log_loss,
                'accuracy': train_accuracy,
                'n_features': X_train.shape[1]
            }

        except Exception as e:
            print(f"{name:<20} {'ERROR':<12} {'ERROR':<12} {X_train.shape[1]:<10,}")
            results[name] = {'error': str(e)}

    print("=" * 70)
    print("📈 Note: Training metrics shown. Use cross-validation for unbiased estimates.")

    return results


def select_best_models(models, cv_results=None, top_k=3):
    """
    Select the best performing models for ensembling.

    Args:
        models: Dictionary of trained models
        cv_results: Cross-validation results (optional)
        top_k: Number of top models to select

    Returns:
        dict: Selected models
    """
    if cv_results is None:
        # If no CV results, return all models
        return models

    # Sort models by log-loss performance
    model_scores = [(name, results.get('log_loss', float('inf')))
                    for name, results in cv_results.items()
                    if 'log_loss' in results]

    model_scores.sort(key=lambda x: x[1])  # Sort by log-loss (lower is better)

    # Select top k models
    selected_names = [name for name, _ in model_scores[:top_k]]
    selected_models = {name: models[name] for name in selected_names if name in models}

    print(f"\n🏆 Selected top {len(selected_models)} models for ensembling:")
    for i, (name, score) in enumerate(model_scores[:top_k]):
        print(f"   {i + 1}. {name}: {score:.4f} log-loss")

    return selected_models