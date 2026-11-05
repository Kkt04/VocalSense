"""
Phase 4 Deep Learning Speech Emotion Recognition Training & Comparative Benchmark.
Executes:
1. Feature extraction & caching for:
   - 43 Acoustic Features (MFCC, F0, RMS, Spectral, Pauses)
   - 768-dim Wav2Vec2 Contextual Speech Embeddings (facebook/wav2vec2-base)
   - 768-dim HuBERT Masked Prediction Embeddings (facebook/hubert-base-ls960)
2. 5-Fold GroupKFold Cross-Validation (strictly grouped by actor_id) across all 4 models:
   - SVM Baseline
   - Random Forest Baseline
   - Wav2Vec2 Classifier
   - HuBERT Classifier
3. Full dataset fit and serialization of production artifacts in backend/saved_models/phase4/
4. MongoDB metadata persistence.
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from backend.ml.feature_extraction import FEATURE_NAMES
from backend.app.services.emotion_recognition.wav2vec2_model import Wav2Vec2Pipeline
from backend.app.services.emotion_recognition.hubert_model import HuBERTPipeline
from backend.app.services.emotion_recognition.deep_emotion_classifier import DeepEmotionClassifier
from backend.app.db.mongodb import mongodb_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_deep_ser")

MANIFEST_PATH = "data/ravdess/metadata.csv"
ACOUSTIC_FEATURES_CSV = "data/ravdess/features_ravdess.csv"
EMBEDDINGS_DIR = "data/embeddings"
OUTPUT_MODELS_DIR = "backend/saved_models/phase4"


def load_dataset():
    df = pd.read_csv(MANIFEST_PATH)
    logger.info("Loaded RAVDESS manifest with %d recordings and %d actors.", len(df), df["actor_id"].nunique())
    classes = sorted(df["emotion_label"].unique().tolist())
    y = df["emotion_label"].values
    groups = df["actor_id"].values
    return df, y, groups, classes


def get_or_extract_acoustic_features(df: pd.DataFrame) -> np.ndarray:
    if os.path.exists(ACOUSTIC_FEATURES_CSV):
        logger.info("Loading cached acoustic features from %s...", ACOUSTIC_FEATURES_CSV)
        df_feats = pd.read_csv(ACOUSTIC_FEATURES_CSV)
        return df_feats[FEATURE_NAMES].values
    raise FileNotFoundError(f"Acoustic features not found at {ACOUSTIC_FEATURES_CSV}")


def extract_embeddings_in_batches(
    pipeline: Any,
    file_paths: List[str],
    batch_size: int = 8,
    model_name: str = "model",
) -> np.ndarray:
    """Extracts embeddings using batch inference with padding and active cache eviction."""
    all_embeddings: List[np.ndarray] = []
    total = len(file_paths)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch_paths = file_paths[i : i + batch_size]
        batch_signals = [pipeline.preprocess_audio(p) for p in batch_paths]

        inputs = pipeline.feature_extractor(
            batch_signals,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(pipeline.device)

        with torch.no_grad():
            outputs = pipeline.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            all_embeddings.append(emb)

        # Free GPU / MPS memory buffers after every batch
        del inputs, outputs
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        done = min(i + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed
        if done % 80 == 0 or done == total:
            logger.info("[%s] Extracted %d/%d (%.1f%%) — %.1f samples/sec", model_name, done, total, 100 * done / total, rate)

    return np.vstack(all_embeddings)


def get_or_extract_deep_embeddings(
    model_type: str,
    df: pd.DataFrame,
) -> np.ndarray:
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    cache_path = os.path.join(EMBEDDINGS_DIR, f"{model_type}_ravdess.npz")

    if os.path.exists(cache_path):
        logger.info("Loading cached %s embeddings from %s...", model_type, cache_path)
        data = np.load(cache_path)
        return data["embeddings"]

    logger.info("Initializing %s pipeline...", model_type)
    if model_type == "wav2vec2":
        pipeline = Wav2Vec2Pipeline()
    elif model_type == "hubert":
        pipeline = HuBERTPipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info("Extracting %s embeddings across %d recordings...", model_type, len(df))
    X = extract_embeddings_in_batches(
        pipeline=pipeline,
        file_paths=df["file_path"].tolist(),
        batch_size=8,
        model_name=model_type,
    )

    np.savez_compressed(
        cache_path,
        embeddings=X,
        labels=df["emotion_label"].values,
        actor_ids=df["actor_id"].values,
        filenames=df["filename"].values,
    )
    logger.info("Cached %s embeddings to %s (shape: %s, size: %.2f MB)", model_type, cache_path, X.shape, os.path.getsize(cache_path) / (1024 * 1024))
    return X


def evaluate_model_speaker_independent(
    name: str,
    create_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    gkf = GroupKFold(n_splits=5)
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    fold_accuracies: List[float] = []
    fold_macro_f1s: List[float] = []

    t0 = time.time()
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = create_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        fold_accuracies.append(float(acc))
        fold_macro_f1s.append(float(f1))

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    eval_time = time.time() - t0

    acc = float(accuracy_score(y_true_all, y_pred_all))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="weighted", zero_division=0
    )

    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=classes, zero_division=0
    )
    per_class = {
        cls_name: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for cls_name, p, r, f, s in zip(classes, class_p, class_r, class_f1, class_supp)
    }

    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)

    logger.info(
        "[%s] 5-Fold GroupKFold Result — Acc: %.2f%% | Macro-F1: %.4f | Weighted-F1: %.4f (Time: %.2fs)",
        name,
        acc * 100,
        f1_macro,
        f1_weight,
        eval_time,
    )

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "fold_macro_f1s": [round(f, 4) for f in fold_macro_f1s],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "eval_time_sec": round(eval_time, 2),
        "sample_count": len(y_true_all),
    }


async def persist_metadata_to_mongodb(results: Dict[str, Any]):
    logger.info("Persisting model metadata to MongoDB collection 'model_metadata'...")
    for model_key, res in results.items():
        doc = {
            "model_name": res["model_name"],
            "model_version": "1.0.0",
            "model_type": model_key,
            "supported_languages": ["en (RAVDESS Speech Benchmark)"],
            "training_dataset": "RAVDESS (1,440 recordings, 24 actors, 8 emotions)",
            "evaluation_metrics": {
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "macro_precision": res["macro_precision"],
                "macro_recall": res["macro_recall"],
                "fold_accuracies": res["fold_accuracies"],
            },
        }
        await mongodb_manager.save_model_metadata(doc)
    logger.info("MongoDB model metadata successfully saved.")


def main():
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    df, y, groups, classes = load_dataset()

    # 1. Acoustic Features & Baselines
    X_acoustic = get_or_extract_acoustic_features(df)

    logger.info("--- Evaluating Traditional SVM Baseline ---")
    svm_res = evaluate_model_speaker_independent(
        name="Support Vector Machine",
        create_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=2.5, probability=True, random_state=42)),
        ]),
        X=X_acoustic,
        y=y,
        groups=groups,
        classes=classes,
    )
    svm_res["feature_representation"] = "43 Acoustic Features (MFCC, F0, RMS, Spectral, Pauses)"

    logger.info("--- Evaluating Random Forest Baseline ---")
    rf_res = evaluate_model_speaker_independent(
        name="Random Forest",
        create_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=180, max_depth=14, random_state=42, n_jobs=-1)),
        ]),
        X=X_acoustic,
        y=y,
        groups=groups,
        classes=classes,
    )
    rf_res["feature_representation"] = "43 Acoustic Features"

    # 2. Wav2Vec2 Deep Speech Embeddings
    logger.info("--- Extracting / Loading Wav2Vec2 Embeddings ---")
    X_w2v = get_or_extract_deep_embeddings("wav2vec2", df)

    logger.info("--- Evaluating Wav2Vec2 Downstream Classifier ---")
    w2v_res = evaluate_model_speaker_independent(
        name="Wav2Vec2 (facebook/wav2vec2-base)",
        create_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        X=X_w2v,
        y=y,
        groups=groups,
        classes=classes,
    )
    w2v_res["feature_representation"] = "768 Deep Speech Embeddings (Wav2Vec2 Base)"

    # Fit and save final Wav2Vec2 production classifier on full dataset
    w2v_classifier = DeepEmotionClassifier(model_type="wav2vec2", classifier_type="logistic_regression")
    w2v_classifier.fit(X_w2v, y)
    w2v_classifier.save(
        model_path=os.path.join(OUTPUT_MODELS_DIR, "wav2vec2_emotion_classifier.joblib"),
        metadata_path=os.path.join(OUTPUT_MODELS_DIR, "wav2vec2_metadata.json"),
        extra_metadata={
            "accuracy": w2v_res["accuracy"],
            "macro_f1": w2v_res["macro_f1"],
            "weighted_f1": w2v_res["weighted_f1"],
            "checkpoint": "facebook/wav2vec2-base",
            "hidden_size": 768,
        },
    )

    # 3. HuBERT Deep Speech Embeddings
    logger.info("--- Extracting / Loading HuBERT Embeddings ---")
    X_hubert = get_or_extract_deep_embeddings("hubert", df)

    logger.info("--- Evaluating HuBERT Downstream Classifier ---")
    hubert_res = evaluate_model_speaker_independent(
        name="HuBERT (facebook/hubert-base-ls960)",
        create_fn=lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        X=X_hubert,
        y=y,
        groups=groups,
        classes=classes,
    )
    hubert_res["feature_representation"] = "768 Deep Speech Embeddings (HuBERT Base)"

    # Fit and save final HuBERT production classifier on full dataset
    hubert_classifier = DeepEmotionClassifier(model_type="hubert", classifier_type="logistic_regression")
    hubert_classifier.fit(X_hubert, y)
    hubert_classifier.save(
        model_path=os.path.join(OUTPUT_MODELS_DIR, "hubert_emotion_classifier.joblib"),
        metadata_path=os.path.join(OUTPUT_MODELS_DIR, "hubert_metadata.json"),
        extra_metadata={
            "accuracy": hubert_res["accuracy"],
            "macro_f1": hubert_res["macro_f1"],
            "weighted_f1": hubert_res["weighted_f1"],
            "checkpoint": "facebook/hubert-base-ls960",
            "hidden_size": 768,
        },
    )

    # 4. Save Comprehensive Comparison Results
    all_results = {
        "svm_baseline": svm_res,
        "random_forest": rf_res,
        "wav2vec2": w2v_res,
        "hubert": hubert_res,
    }
    comparison_path = os.path.join(OUTPUT_MODELS_DIR, "model_comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved complete comparison report to %s", comparison_path)

    # 5. Persist to MongoDB
    asyncio.run(persist_metadata_to_mongodb(all_results))

    print("\n" + "=" * 80)
    print("PHASE 4 BENCHMARK COMPARISON COMPLETE (SPEAKER-INDEPENDENT 5-FOLD GROUPKFOLD)")
    print("=" * 80)
    print(f"{'Model':<35} | {'Features':<22} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Weighted-F1':<10}")
    print("-" * 95)
    for k, v in all_results.items():
        feat = "43 Acoustic" if "baseline" in k or "random" in k else "768 Deep Embeds"
        print(f"{v['model_name']:<35} | {feat:<22} | {v['accuracy']*100:>6.2f}%    | {v['macro_f1']:>8.4f}   | {v['weighted_f1']:>8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    from typing import Tuple
    main()
