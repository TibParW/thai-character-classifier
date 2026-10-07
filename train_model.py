import sys
from pathlib import Path
import time
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocessing import image_to_features
from labels import LABEL_TO_DISPLAY

# Ensure UTF-8 output in Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "thai_characters"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "thai_character_classifier.joblib"


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(data_dir):
    """
    Load Thai character images from directory structure:
        thai_characters/
            01_kor_kai/
            02_khor_khai/
            ...
            44_hor_nokhuk/
    Folder name is used as the class label.
    """
    X = []
    y = []

    supported_extensions = {".jpg", ".jpeg", ".png"}
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"Found {len(class_dirs)} character classes in {data_dir.name}")

    for class_dir in class_dirs:
        label = class_dir.name
        display_name = LABEL_TO_DISPLAY.get(label, label)
        image_files = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in supported_extensions]
        )
        print(f"  Loading class {label} ({display_name}): {len(image_files)} images...")

        for image_path in image_files:
            try:
                features = image_to_features(image_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"    [Warning] Could not load {image_path}: {e}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    return X, y


# ============================================================
# Main Training Routine
# ============================================================

def main():
    print("=" * 70)
    print("Thai Consonant Classifier Training (พยัญชนะไทย 44 ตัว ก - ฮ)")
    print("=" * 70)

    # 1. Load dataset
    start_time = time.time()
    X, y = load_dataset(DATA_DIR)
    load_time = time.time() - start_time

    print()
    print("Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape} (28x28 = 784 pixels per sample)")
    print(f"  Pixel range: min = {X.min():.2f}, max = {X.max():.2f}")
    print(f"  Loading and feature extraction time: {load_time:.2f} seconds")

    # 2. Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nClass balance check: {len(unique_labels)} classes, {counts.min()} to {counts.max()} images/class")

    # 3. Train/Test split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(f"Data Splitting (80/20 Stratified):")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")

    # 4. Train Model 1: Random Forest Classifier
    print()
    print("-" * 70)
    print("Training Model 1: Random Forest Classifier (n_estimators=200, n_jobs=-1)...")
    print("-" * 70)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro")
    f1_weighted_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print(f"Random Forest Training Completed in {rf_train_time:.2f}s")
    print(f"  Test Accuracy:     {acc_rf * 100:.2f}%")
    print(f"  Macro-averaged F1: {f1_macro_rf:.4f}")
    print(f"  Weighted F1:       {f1_weighted_rf:.4f}")

    # 5. Train Model 2: SVM (RBF Kernel) with StandardScaler
    print()
    print("-" * 70)
    print("Training Model 2: Support Vector Machine (RBF Kernel, C=10)...")
    print("-" * 70)

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])

    t0 = time.time()
    svm_model.fit(X_train, y_train)
    svm_train_time = time.time() - t0

    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_macro_svm = f1_score(y_test, y_pred_svm, average="macro")
    f1_weighted_svm = f1_score(y_test, y_pred_svm, average="weighted")

    print(f"SVM Training Completed in {svm_train_time:.2f}s")
    print(f"  Test Accuracy:     {acc_svm * 100:.2f}%")
    print(f"  Macro-averaged F1: {f1_macro_svm:.4f}")
    print(f"  Weighted F1:       {f1_weighted_svm:.4f}")

    # 6. Model Comparison and Selection
    print()
    print("=" * 70)
    print("Model Comparison Summary:")
    print("=" * 70)
    print(f"  Random Forest: Accuracy = {acc_rf * 100:.2f}%, F1 = {f1_macro_rf:.4f}, Train Time = {rf_train_time:.2f}s")
    print(f"  SVM (RBF):     Accuracy = {acc_svm * 100:.2f}%, F1 = {f1_macro_svm:.4f}, Train Time = {svm_train_time:.2f}s")

    # Select best model
    if acc_rf >= acc_svm:
        best_model = rf_model
        best_name = "Random Forest"
        best_pred = y_pred_rf
        best_acc = acc_rf
    else:
        best_model = svm_model
        best_name = "SVM (RBF)"
        best_pred = y_pred_svm
        best_acc = acc_svm

    print(f"\n=> Best Model Selected: {best_name} (Accuracy: {best_acc * 100:.2f}%)")

    # 7. Detailed Evaluation of Selected Best Model
    print()
    print("=" * 70)
    print(f"Detailed Classification Report ({best_name})")
    print("=" * 70)
    # Generate readable target names
    target_names = [LABEL_TO_DISPLAY.get(lbl, lbl) for lbl in sorted(unique_labels)]
    print(classification_report(y_test, best_pred, target_names=target_names, digits=4))

    # Confusion Analysis: Top confused pairs
    print("-" * 70)
    print("Error Analysis (Top Confused Character Pairs):")
    print("-" * 70)
    conf_pairs = {}
    for true_lbl, pred_lbl in zip(y_test, best_pred):
        if true_lbl != pred_lbl:
            pair = (LABEL_TO_DISPLAY.get(true_lbl, true_lbl), LABEL_TO_DISPLAY.get(pred_lbl, pred_lbl))
            conf_pairs[pair] = conf_pairs.get(pair, 0) + 1

    if conf_pairs:
        sorted_conf = sorted(conf_pairs.items(), key=lambda x: x[1], reverse=True)
        for (true_ch, pred_ch), count in sorted_conf[:10]:
            print(f"  True: {true_ch:<15s} misclassified as -> {pred_ch:<15s} ({count} times)")
    else:
        print("  No classification errors detected on test set!")

    # 8. Save Trained Model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    print()
    print("=" * 70)
    print(f"Model saved successfully to: {MODEL_PATH}")
    print(f"File size: {model_size_mb:.2f} MB (Optimized for GitHub & Web Deployment)")
    print("=" * 70)

    # 9. Verify Reloaded Model
    print("\nVerifying reloaded model on test sample...")
    reloaded = joblib.load(MODEL_PATH)
    sample_preds = reloaded.predict(X_test[:5])
    print(f"True labels:      {[LABEL_TO_DISPLAY.get(l, l) for l in y_test[:5]]}")
    print(f"Predicted labels: {[LABEL_TO_DISPLAY.get(l, l) for l in sample_preds]}")
    print("Verification complete! Ready for Gradio Web App.")


if __name__ == "__main__":
    main()
