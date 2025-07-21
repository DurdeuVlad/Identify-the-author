# --- main.py ---
"""
Entry point for the Identify the Author pipeline.
Loads data, preprocesses text, engineers features, trains models,
performs evaluation, diagnostics, builds ensemble, and writes submission.
"""
import os
from data_loading import load_data
from text_preprocessing import preprocess_dataframe
from feature_engineering import build_features
from model_training import train_models
from evaluation import evaluate_models, calibrate_model
from diagnostics import (
    plot_learning_curve,
    holdout_evaluation,
    plot_calibration_curve,
    print_confusion_report
)
from ensembling import ensemble_predictions
from submission import make_submission


def main(data_dir: str, output_dir: str):
    # 1. Load data
    train_df, test_df, sample_sub_df = load_data(data_dir)

    # 2. Preprocess text
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # 3. Engineer features
    X_train, X_test, feature_info = build_features(train_df, test_df)
    y = train_df.author.values

    # 4. Train individual models
    models, predictions = train_models(X_train, y)

    # 5. Evaluate and calibrate
    evaluate_models(models, X_train, y)
    calibrated_models = {
        name: calibrate_model(m, X_train, y)
        for name, m in models.items()
    }

    # 6. Diagnostics on baseline model (e.g., Logistic Regression)
    baseline = calibrated_models.get('lr')
    if baseline:
        plot_learning_curve(baseline, X_train, y)
        holdout_evaluation(baseline, X_train, y)
        # for each class index
        for idx in range(len(baseline.classes_)):
            plot_calibration_curve(baseline, X_train, y, class_index=idx)
        # use holdout split for confusion matrix
        X_tr, X_val, y_tr, y_val = baseline._cv.split(X_train, y) if hasattr(baseline, '_cv') else None
        if X_val is not None:
            print_confusion_report(baseline, X_val, y_val)

    # 7. Ensemble predictions (use calibrated models' predictions)
    cal_preds = {name: model.predict_proba(X_train)
                 for name, model in calibrated_models.items()}
    final_probas = ensemble_predictions(cal_preds)

    # 8. Submission
    os.makedirs(output_dir, exist_ok=True)
    make_submission(
        ids=test_df.id.values,
        probas=final_probas,
        sample_submission_df=sample_sub_df,
        output_path=os.path.join(output_dir, 'submission.csv')
    )


if __name__ == '__main__':
    DATA_DIR = './data'
    OUTPUT_DIR = './outputs'
    main(DATA_DIR, OUTPUT_DIR)