#!/usr/bin/env python3
# src/run_pipeline.py - TensorFlow-Free Version
"""
Complete pipeline runner for the Identify the Author competition.
NO TENSORFLOW - Uses PyTorch + XGBoost + Scikit-learn stack.
"""
import os
import sys
import traceback
from datetime import datetime


def check_tf_free_environment():
    """Check that we have a clean TensorFlow-free environment."""
    print("🚀 Checking TensorFlow-free environment...")

    # Check core packages
    required_core = ['numpy', 'pandas', 'sklearn', 'scipy']
    optional_packages = ['xgboost', 'torch', 'transformers']

    missing_core = []
    missing_optional = []
    tensorflow_detected = False

    # Check core packages
    for package in required_core:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_core.append(package)
            print(f"   ❌ {package} (REQUIRED)")

    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"   ⚠️  {package} (optional - will use fallbacks)")

    # Check for TensorFlow (should NOT be present)
    try:
        import tensorflow
        tensorflow_detected = True
        print(f"   ⚠️  tensorflow detected (not needed for this pipeline)")
    except ImportError:
        print(f"   ✅ No tensorflow (clean environment)")

    # Environment assessment
    if missing_core:
        print(f"\n❌ Missing REQUIRED packages: {missing_core}")
        print("Install with: pip install numpy pandas scikit-learn scipy")
        return False

    if missing_optional:
        print(f"\n📦 Optional packages missing: {missing_optional}")
        print("For full features: pip install xgboost torch transformers")
        print("Pipeline will work with fallbacks")

    if tensorflow_detected:
        print(f"\n⚡ TensorFlow detected but not used (clean separation)")

    print("\n✅ Environment ready for TensorFlow-free pipeline!")
    return True


def setup_environment():
    """Setup the environment and check dependencies."""
    print("🔧 Setting up TensorFlow-free ML environment...")

    if not check_tf_free_environment():
        return False

    # Set random seeds for reproducibility
    import numpy as np
    np.random.seed(42)

    # Configure warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("✅ Environment configured successfully")
    return True


def run_tf_free_pipeline():
    """Run the complete TensorFlow-free ML pipeline."""
    print("\n" + "=" * 80)
    print("🎯 IDENTIFY THE AUTHOR - TENSORFLOW-FREE PIPELINE")
    print("🚫 NO TENSORFLOW | ✅ PYTORCH + XGBOOST + SCIKIT-LEARN")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not setup_environment():
        print("❌ Environment setup failed")
        return False

    try:
        # Import our clean modules
        print("\n📦 Importing TensorFlow-free modules...")
        from data_loading import load_data
        from text_preprocessing import preprocess_dataframe
        from feature_engineering import build_features
        from model_training import train_models, evaluate_model_cv, get_model_performance_summary
        from evaluation import evaluate_models
        from ensembling import create_production_ensemble
        from submission import generate_multiple_submissions

        print("✅ All modules imported successfully (no TensorFlow)")

        # Configuration
        DATA_DIR = './data'
        OUTPUT_DIR = './outputs'
        CV_FOLDS = 5

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Stage 1: Data Loading
        print(f"\n{'=' * 20} STAGE 1: DATA LOADING {'=' * 20}")
        train_df, test_df, sample_sub_df = load_data(DATA_DIR)

        # Stage 2: Text Preprocessing
        print(f"\n{'=' * 20} STAGE 2: PREPROCESSING {'=' * 20}")
        train_df = preprocess_dataframe(train_df)
        test_df = preprocess_dataframe(test_df)

        # Stage 3: Feature Engineering
        print(f"\n{'=' * 20} STAGE 3: FEATURE ENGINEERING {'=' * 20}")
        X_train, X_test, feature_info = build_features(train_df, test_df)
        y_train = train_df['author'].values

        # Stage 4: Model Training (TensorFlow-free)
        print(f"\n{'=' * 20} STAGE 4: MODEL TRAINING (NO TF) {'=' * 20}")
        models = train_models(X_train, y_train, feature_type='tfidf')

        # Stage 5: Model Evaluation
        print(f"\n{'=' * 20} STAGE 5: EVALUATION {'=' * 20}")
        cv_results = {}
        for name, model in models.items():
            print(f"\n🔍 Evaluating {name}...")
            cv_result = evaluate_model_cv(model, X_train, y_train, cv=CV_FOLDS)
            cv_results[name] = cv_result

            log_loss_mean = cv_result['log_loss']
            log_loss_std = cv_result['log_loss_std']
            accuracy_mean = cv_result['accuracy']

            print(f"   Log-loss: {log_loss_mean:.4f} ± {log_loss_std:.4f}")
            print(f"   Accuracy: {accuracy_mean:.3f}")

        # Stage 6: Model Performance Summary
        print(f"\n{'=' * 20} STAGE 6: PERFORMANCE SUMMARY {'=' * 20}")
        performance_summary = get_model_performance_summary(models, X_train, y_train)

        # Find best model by log-loss
        best_model_name = min(cv_results.keys(),
                              key=lambda x: cv_results[x]['log_loss'])
        best_log_loss = cv_results[best_model_name]['log_loss']

        print(f"\n🏆 Best single model: {best_model_name} ({best_log_loss:.4f} log-loss)")

        # Stage 7: Ensemble Creation
        print(f"\n{'=' * 20} STAGE 7: ENSEMBLE CREATION {'=' * 20}")
        ensemble = create_production_ensemble(models, cv_results)

        # Stage 8: Submission Generation
        print(f"\n{'=' * 20} STAGE 8: SUBMISSION GENERATION {'=' * 20}")
        test_ids = test_df['id'].values

        # Generate submissions for all models
        submission_files = generate_multiple_submissions(
            models, X_test, test_ids, sample_sub_df, OUTPUT_DIR
        )

        # Generate ensemble submission
        print(f"\n📤 Generating ensemble submission...")
        ensemble_probs = ensemble.predict_proba(X_test)

        from submission import make_submission
        ensemble_path = os.path.join(OUTPUT_DIR, 'submission_ensemble_optimized.csv')
        make_submission(test_ids, ensemble_probs, sample_sub_df, ensemble_path)

        # Final summary
        print(f"\n{'=' * 20} PIPELINE COMPLETE {'=' * 20}")
        print(f"✅ Pipeline completed successfully!")
        print(f"🎯 Best single model: {best_model_name} ({best_log_loss:.4f} log-loss)")
        print(f"📁 Submissions saved to: {OUTPUT_DIR}")
        print(f"🚫 No TensorFlow used - clean PyTorch/XGBoost/sklearn stack")

        # List generated files
        print(f"\n📋 Generated submission files:")
        for name, path in submission_files.items():
            print(f"   • {name}: {path}")
        print(f"   • ensemble_optimized: {ensemble_path}")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def quick_test():
    """Quick test of core functionality without full pipeline."""
    print("🧪 Quick TensorFlow-free functionality test...")

    try:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create dummy data
        texts = ["Upon a midnight dreary", "The Old Ones were", "Frankenstein's monster"]
        labels = ["EAP", "HPL", "MWS"]

        # Basic TF-IDF + LogReg test
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)

        model = LogisticRegression(random_state=42)
        model.fit(X, labels)

        probs = model.predict_proba(X)

        print("✅ Core functionality test passed")
        print(f"   TF-IDF shape: {X.shape}")
        print(f"   Predictions shape: {probs.shape}")
        print(f"   Classes: {model.classes_}")

        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Allow command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            success = quick_test()
        elif sys.argv[1] == "--env":
            success = check_tf_free_environment()
        else:
            print("Usage: python run_pipeline.py [--test|--env]")
            success = False
    else:
        success = run_tf_free_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)