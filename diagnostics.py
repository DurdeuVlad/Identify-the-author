# --- diagnostics.py ---
"""
Diagnostic functions to assess model performance before final submission.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import log_loss, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

def plot_learning_curve(model, X, y):
    """
    Plots training and validation log-loss vs. training set size.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_log_loss',
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0], shuffle=True, random_state=42
    )
    train_loss = -np.mean(train_scores, axis=1)
    val_loss   = -np.mean(val_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_loss, label='Train')
    plt.plot(train_sizes, val_loss,   label='Validation')
    plt.xlabel('Training set size')
    plt.ylabel('Log-Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()

def holdout_evaluation(model, X, y, test_size=0.1, random_state=42):
    """
    Splits data into train/hold-out, fits model, prints hold-out log-loss.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)
    loss = log_loss(y_val, probs)
    print(f"Hold-out log-loss: {loss:.4f}")
    return X_val, y_val  # return for further diagnostics

def plot_calibration_curve(model, X, y, class_index=0, n_bins=10):
    """
    Plots calibration curve for a specified class.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)[:, class_index]
    true = (y_val == model.classes_[class_index]).astype(int)
    frac_pos, mean_pred = calibration_curve(true, probs, n_bins=n_bins)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.plot([0,1],[0,1],'--')  # perfect calibration
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve: class {model.classes_[class_index]}')
    plt.show()

def print_confusion_report(model, X_val, y_val):
    """
    Prints confusion matrix and classification report on hold-out data.
    """
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print('Confusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_val, y_pred))
