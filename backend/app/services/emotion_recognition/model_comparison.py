"""
Rigorous Speaker-Independent Model Comparison and Evaluation Suite.
Compares:
1. Support Vector Machine (Traditional 43 Acoustic Features)
2. Random Forest (Traditional 43 Acoustic Features)
3. Wav2Vec2 (768 Deep Speech Embeddings)
4. HuBERT (768 Deep Speech Embeddings)

All evaluated under identical conditions:
- Dataset: RAVDESS (1,440 recordings, 24 actors, 8 emotions)
- Protocol: 5-Fold GroupKFold grouped strictly by actor_id
- Zero speaker leakage: No actor appears in both train and test partitions.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def evaluate_speaker_independent_cv(
    model_name: str,
    create_pipeline_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: List[str],
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates a model pipeline using 5-Fold GroupKFold grouped by actor_id.
    Accumulates out-of-fold predictions to prevent speaker leakage.
    """
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    fold_accuracies: List[float] = []
    fold_macro_f1s: List[float] = []

    start_time = time.time()
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = create_pipeline_fn()
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)

        fold_accuracies.append(float(acc))
        fold_macro_f1s.append(float(f1))
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    total_time = time.time() - start_time

    # Aggregate overall metrics across the entire dataset out-of-fold
    acc = float(accuracy_score(y_true_all, y_pred_all))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="weighted", zero_division=0
    )

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=classes, zero_division=0
    )
    per_class_metrics = {
        cls_name: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for cls_name, p, r, f, s in zip(classes, class_p, class_r, class_f1, class_supp)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "fold_macro_f1s": [round(f, 4) for f in fold_macro_f1s],
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "total_eval_time_sec": round(total_time, 2),
        "avg_latency_ms": round((total_time / len(y_true_all)) * 1000, 2),
    }


def run_full_benchmark_comparison(
    manifest_csv: str = "data/ravdess/metadata.csv",
    acoustic_features_file: str = "data/ravdess/acoustic_features.npy",
    output_dir: str = "backend/saved_models/phase4",
) -> Dict[str, Any]:
    """
    Executes the comparative experiment across all 4 models:
    SVM (43 features), RF (43 features), Wav2Vec2 (768), HuBERT (768).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    classes = sorted(df["emotion_label"].unique().tolist())
    y = df["emotion_label"].values
    groups = df["actor_id"].values

    results: Dict[str, Any] = {}

    # 1. Acoustic Features (Phase 3 Baseline)
    from backend.ml.feature_extraction import extract_features_batch, FEATURE_NAMES
    if os.path.exists(acoustic_features_file):
        X_acoustic = np.load(acoustic_features_file)
    else:
        logger.info("Extracting acoustic features for baseline...")
        X_acoustic = extract_features_batch(df["file_path"].tolist())
        np.save(acoustic_features_file, X_acoustic)

    logger.info("Evaluating Traditional SVM Baseline...")
    svm_res = evaluate_speaker_independent_cv(
        model_name="Support Vector Machine (Phase 3 Baseline)",
        create_pipeline_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=2.5, probability=True, random_state=42)),
        ]),
        X=X_acoustic,
        y=y,
        groups=groups,
        classes=classes,
    )
    svm_res["feature_representation"] = "43 Acoustic Features (MFCC, F0, RMS, Spectral, Pauses)"
    results["svm_baseline"] = svm_res

    logger.info("Evaluating Random Forest Baseline...")
    rf_res = evaluate_speaker_independent_cv(
        model_name="Random Forest",
        create_pipeline_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=180, max_depth=14, random_state=42, n_jobs=-1)),
        ]),
        X=X_acoustic,
        y=y,
        groups=groups,
        classes=classes,
    )
    rf_res["feature_representation"] = "43 Acoustic Features"
    results["random_forest"] = rf_res

    # 2. Wav2Vec2 Representations
    from backend.app.services.emotion_recognition.embedding_extractor import DatasetEmbeddingExtractor
    extractor = DatasetEmbeddingExtractor()

    logger.info("Extracting/Loading Wav2Vec2 Embeddings...")
    w2v_data = extractor.extract_or_load("wav2vec2", df)
    X_w2v = w2v_data["embeddings"]

    logger.info("Evaluating Wav2Vec2 Downstream Classifier...")
    w2v_res = evaluate_speaker_independent_cv(
        model_name="Wav2Vec2 (facebook/wav2vec2-base)",
        create_pipeline_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        X=X_w2v,
        y=y,
        groups=groups,
        classes=classes,
    )
    w2v_res["feature_representation"] = "768 Deep Speech Embeddings (Wav2Vec2)"
    results["wav2vec2"] = w2v_res

    # 3. HuBERT Representations
    logger.info("Extracting/Loading HuBERT Embeddings...")
    hubert_data = extractor.extract_or_load("hubert", df)
    X_hubert = hubert_data["embeddings"]

    logger.info("Evaluating HuBERT Downstream Classifier...")
    hubert_res = evaluate_speaker_independent_cv(
        model_name="HuBERT (facebook/hubert-base-ls960)",
        create_pipeline_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        X=X_hubert,
        y=y,
        groups=groups,
        classes=classes,
    )
    hubert_res["feature_representation"] = "768 Deep Speech Embeddings (HuBERT)"
    results["hubert"] = hubert_res

    # Save summary report
    output_path = os.path.join(output_dir, "model_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved complete model comparison results to %s", output_path)

    return results
