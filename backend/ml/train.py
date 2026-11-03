"""
Model Training and Speaker-Independent Evaluation for Speech Emotion Recognition.
Compares:
1. Logistic Regression
2. Support Vector Machine (SVM with RBF kernel)
3. Random Forest Classifier

Implements:
- Primary: Speaker-Independent GroupKFold (grouped by actor_id)
- Secondary: StratifiedKFold (to demonstrate speaker leakage)
- Metric logging: Accuracy, Precision, Recall, Macro-F1, Weighted-F1
- Confusion Matrix visualization
- Model artifact serialization with metadata
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from backend.ml.feature_extraction import FEATURE_NAMES

SAVED_MODELS_DIR = "backend/saved_models"
PLOTS_DIR = "backend/ml/eda_plots"


def create_models() -> Dict[str, Any]:
    """Define the 3 candidate model pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=2.5,
                gamma="scale",
                probability=True,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=180,
                max_depth=14,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_split_generator,
    groups: Optional[np.ndarray] = None,
    classes: List[str] = None,
) -> Dict[str, Any]:
    """
    Run k-fold cross validation and accumulate predictions across out-of-fold splits.
    """
    y_true_all = []
    y_pred_all = []

    splits = cv_split_generator.split(X, y, groups=groups)

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit on training fold
        pipeline.fit(X_train, y_train)
        # Predict on holdout fold
        preds = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Compute aggregate metrics
    acc = accuracy_score(y_true_all, y_pred_all)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="weighted", zero_division=0
    )
    conf_mat = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    cls_report = classification_report(y_true_all, y_pred_all, labels=classes, zero_division=0, output_dict=True)

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": cls_report,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    classes: List[str],
    model_name: str,
    output_path: str,
) -> None:
    """Plot and save normalized confusion matrix."""
    # Normalize by row (actual class)
    conf_norm = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    conf_norm = np.nan_to_num(conf_norm)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        conf_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes],
        cbar_kws={"label": "Normalized Accuracy"},
    )
    plt.title(f"Confusion Matrix: {model_name}\n(Speaker-Independent GroupKFold)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=10)
    plt.ylabel("True Emotion", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot: {output_path}")


def train_and_compare_models(
    features_csv: str = "data/ravdess/features_ravdess.csv",
    saved_models_dir: str = SAVED_MODELS_DIR,
    plots_dir: str = PLOTS_DIR,
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression, SVM, and Random Forest models.
    Performs speaker-independent cross-validation and saves the top performer.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(features_csv)

    # Features and Targets
    X = df[FEATURE_NAMES].values
    y = df["emotion_label"].values
    actor_groups = df["actor_id"].values
    classes = sorted(list(df["emotion_label"].unique()))

    print(f"Dataset: {len(df)} samples across {df['actor_id'].nunique()} actors.")
    print(f"Feature dimensionality: {X.shape[1]} acoustic features.")
    print(f"Classes: {classes}\n")

    results_table: List[Dict[str, Any]] = []
    models_dict = create_models()
    trained_pipelines = {}
    eval_details = {}

    # 1. Primary: Speaker-Independent GroupKFold (5 splits on 24 actors)
    print("=" * 65)
    print("PRIMARY EVALUATION: Speaker-Independent GroupKFold (5 splits)")
    print("=" * 65)
    gkf = GroupKFold(n_splits=5)

    for name, pipeline in models_dict.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_cross_validation(
            pipeline=pipeline,
            X=X,
            y=y,
            cv_split_generator=gkf,
            groups=actor_groups,
            classes=classes,
        )

        results_table.append({
            "model": name,
            "split_type": "Speaker-Independent (GroupKFold)",
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        })
        eval_details[name] = metrics

        # Save confusion matrix plot
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), classes, name, cm_path)

        # Retrain full pipeline on 100% of data for production serialization
        pipeline.fit(X, y)
        trained_pipelines[name] = pipeline

    # 2. Secondary: Random Stratified K-Fold (to demonstrate actor leakage)
    print("\n" + "=" * 65)
    print("SECONDARY COMPARISON: Random Stratified K-Fold (Demonstrating Leakage)")
    print("=" * 65)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipeline in create_models().items():
        metrics_leakage = evaluate_cross_validation(
            pipeline=pipeline,
            X=X,
            y=y,
            cv_split_generator=skf,
            classes=classes,
        )
        results_table.append({
            "model": name,
            "split_type": "Random Stratified (Leakage)",
            "accuracy": metrics_leakage["accuracy"],
            "f1_macro": metrics_leakage["f1_macro"],
            "f1_weighted": metrics_leakage["f1_weighted"],
            "precision_macro": metrics_leakage["precision_macro"],
            "recall_macro": metrics_leakage["recall_macro"],
        })

    # Summary table
    df_results = pd.DataFrame(results_table)
    print("\nMODEL PERFORMANCE COMPARISON:")
    print(df_results.to_string(index=False))

    # Select Best Model based on Speaker-Independent Macro-F1
    speaker_indep_results = [r for r in results_table if r["split_type"] == "Speaker-Independent (GroupKFold)"]
    best_record = max(speaker_indep_results, key=lambda x: x["f1_macro"])
    best_name = best_record["model"]
    best_pipeline = trained_pipelines[best_name]

    print(f"\nTop-performing model selected: {best_name} (Macro-F1: {best_record['f1_macro']})")

    # Serialize Best Model Artifacts
    model_path = os.path.join(saved_models_dir, "emotion_classifier.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved model pipeline to {model_path}")

    # Build metadata
    label_mapping = {int(i): label for i, label in enumerate(classes)}
    metadata = {
        "model_name": best_name,
        "algorithm": str(type(best_pipeline.named_steps["classifier"]).__name__),
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "RAVDESS Speech (1440 samples, 24 actors)",
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "classes": classes,
        "label_mapping": label_mapping,
        "evaluation_strategy": "Speaker-Independent 5-fold GroupKFold by actor_id",
        "metrics": {
            "speaker_independent": {
                "accuracy": best_record["accuracy"],
                "f1_macro": best_record["f1_macro"],
                "f1_weighted": best_record["f1_weighted"],
                "precision_macro": best_record["precision_macro"],
                "recall_macro": best_record["recall_macro"],
            },
            "all_models_comparison": results_table,
        },
        "per_class_performance": eval_details[best_name]["classification_report"],
        "confidence_threshold": 0.35,  # Threshold for reliable prediction abstention
    }

    metadata_path = os.path.join(saved_models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate Speech Emotion Recognition models")
    parser.add_argument("--features", default="data/ravdess/features_ravdess.csv")
    args = parser.parse_args()

    train_and_compare_models(features_csv=args.features)
