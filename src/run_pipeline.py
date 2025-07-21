#!/usr/bin/env python3
# --- run_pipeline.py ---
"""
Complete pipeline runner for the Identify the Author competition.
This script runs the entire ML pipeline from data loading to submission generation.
"""
import os
import sys
import traceback
from datetime import datetime


def setup_environment():
    """Setup the environment and check dependencies."""
    print("🚀 Setting up environment...")

    # Check if required modules are available
    required_modules = [
        'numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        print("Please install with: pip install -r requirements.txt")
        return False

    print("✅ All required modules available")
    return True


def run_full_pipeline():
    """Run the complete ML pipeline."""
    print("\n" + "=" * 80)
    print("🎯 IDENTIFY THE AUTHOR - KAGGLE COMPETITION PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Import modules (will use the ones we created)
        print("\n📦 Importing pipeline modules...")
        from data_loading import load_data
        from text_preprocessing import preprocess_dataframe
        from feature_engineering import build_features, get_feature_importance_summary
        from model_training import train_models, get_model_performance_summary
        from evaluation import evaluate_models, calibrate_model, generate_evaluation_report
        from diagnostics import holdout_evaluation, print_confusion_report
        from ensembling import create_advanced_ensemble, create_production_ensemble
        from submission import make_submission, generate_multiple_submissions

        print("✅ All modules imported successfully")

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
        y_train = train_df['author'].values if 'author' in train_df.columns else None

        if y_train is None:
            print("❌ No author labels found in training data")
            return False

        # Display feature summary
        get_feature_importance_summary(feature_info)

        # Stage 4: Model Training
        print(f"\n{'=' * 20} STAGE 4: MODEL TRAINING {'=' * 20}")
        models, train_predictions = train_models(X_train, y_train, cv_folds=CV_FOLDS)

        # Display training performance
        get_model_performance_summary(models, X_train, y_train)

        # Stage 5: Model Evaluation
        print(f"\n{'=' * 20} STAGE 5: MODEL EVALUATION {'=' * 20}")
        evaluation_results = evaluate_models(models, X_train, y_train, cv_folds=CV_FOLDS)

        # Stage 6: Model Calibration
        print(f"\n{'=' * 20} STAGE 6: MODEL CALIBRATION {'=' * 20}")
        calibrated_models = {}
        for name, model in models.items():
            calibrated_models[name] = calibrate_model(model, X_train, y_train)

        # Stage 7: Ensemble Creation
        print(f"\n{'=' * 20} STAGE 7: ENSEMBLE CREATION {'=' * 20}")
        ensemble_config = create_advanced_ensemble(calibrated_models, X_train, y_train)

        if ensemble_config:
            print(f"🏆 Best ensemble strategy: {ensemble_config['best_strategy']}")
            print(f"🎯 Best log-loss: {ensemble_config['best_log_loss']:.6f}")

        # Stage 8: Final Predictions
        print(f"\n{'=' * 20} STAGE 8: FINAL PREDICTIONS {'=' * 20}")

        # Generate test predictions using production ensemble
        final_test_predictions = create_production_ensemble(
            calibrated_models, X_train, y_train, X_test
        )

        if final_test_predictions is not None:
            # Create main submission
            main_submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
            submission_df = make_submission(
                ids=test_df['id'].values,
                probas=final_test_predictions,
                sample_submission_df=sample_sub_df,
                output_path=main_submission_path
            )

            # Generate multiple submissions (individual models + ensemble)
            submission_files = generate_multiple_submissions(
                calibrated_models, X_test, test_df['id'].values,
                sample_sub_df, output_dir=os.path.join(OUTPUT_DIR, 'submissions')
            )

            # Stage 9: Generate Report
            print(f"\n{'=' * 20} STAGE 9: FINAL REPORT {'=' * 20}")
            evaluation_report = generate_evaluation_report(
                models, evaluation_results, X_train, y_train
            )

            # Save report
            import json
            report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
            with open(report_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_report = convert_numpy_types(evaluation_report)
                json.dump(json_report, f, indent=2, default=str)

            print(f"📋 Evaluation report saved to: {report_path}")

        else:
            print("❌ Failed to generate final predictions")
            return False

        # Success summary
        print(f"\n{'=' * 20} PIPELINE COMPLETE {'=' * 20}")
        print(f"✅ Pipeline completed successfully!")
        print(f"📁 Main submission: {main_submission_path}")
        print(f"📂 All outputs in: {OUTPUT_DIR}/")
        print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def run_quick_test():
    """Run a quick test of the pipeline components."""
    print("🧪 Running quick pipeline test...")

    try:
        # Test imports
        from data_loading import load_data
        from text_preprocessing import preprocess_text
        from feature_engineering import build_features
        from model_training import create_logistic_regression_model
        from evaluation import evaluate_models
        from ensembling import weighted_average_ensemble
        from submission import make_submission

        print("✅ All imports successful")

        # Test basic functionality
        sample_text = "This is a test sentence for preprocessing."
        processed = preprocess_text(sample_text)
        print(f"✅ Text preprocessing: '{sample_text}' -> '{processed}'")

        # Test data loading (will create dummy data if files missing)
        train_df, test_df, sample_sub = load_data('./data')
        print(f"✅ Data loading: train={len(train_df)}, test={len(test_df)}")

        print("✅ Quick test passed!")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run quick test only
        success = setup_environment() and run_quick_test()
    else:
        # Run full pipeline
        success = setup_environment() and run_full_pipeline()

    if success:
        print("\n🎉 Success! Pipeline completed without errors.")
        sys.exit(0)
    else:
        print("\n💥 Pipeline failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()