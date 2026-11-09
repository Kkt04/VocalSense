"""
Phase 5 Comprehensive Automated Test Suite.

Covers:
1. Generalisation & Data Leakage Prevention:
   - Speaker-independent GroupKFold partitioning verification.
   - Strict zero-actor-overlap assertion.
   - No duplicate sample leakage across splits.
2. Robustness Perturbations:
   - Additive noise, volume gain scaling, audio truncation.
3. Calibration & Uncertainty Abstention:
   - Calibrated temperature scaling and entropy calculation.
   - Selective abstention gate verification on uncertain/ambiguous signals.
4. Model Explainability:
   - Feature dimensions, feature groups, and permutation consistency.
5. Multilingual Transcription & Language Matching:
   - Hindi Devanagari preservation and Hinglish code-switching detection.
   - Strict language prompt generation and diagnostic guardrails.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from backend.ml.feature_extraction import FEATURE_NAMES
from backend.evaluation.wer_cer import normalize_text_for_eval, compute_metrics
from backend.app.services.language.language_detector import LanguageDetector
from backend.app.services.conversation.language_prompt import build_system_prompt
from backend.app.services.emotion_recognition.deep_emotion_classifier import DeepEmotionClassifier
from experiments.phase5.robustness.run_robustness_experiments import add_noise, apply_gain, truncate_audio


def test_speaker_independent_split_integrity():
    """Verify that GroupKFold strictly prevents speaker leakage across train and test partitions."""
    manifest_path = "data/ravdess/metadata.csv"
    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    df = pd.read_csv(manifest_path)

    actors = df["actor_id"].values
    assert len(df) == 1440
    assert len(np.unique(actors)) == 24

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, groups=actors))
    assert len(splits) == 5

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_actors = set(actors[train_idx])
        test_actors = set(actors[test_idx])

        # Strict zero-leakage guarantee
        overlap = train_actors.intersection(test_actors)
        assert len(overlap) == 0, f"Speaker leakage detected in fold {fold_idx}: {overlap}"

        # Total sample partition guarantee
        assert len(train_idx) + len(test_idx) == len(df)
        assert len(train_idx) > 0 and len(test_idx) > 0


def test_no_duplicate_audio_across_splits():
    """Verify that no filename or file path appears simultaneously in train and test splits."""
    df = pd.read_csv("data/ravdess/metadata.csv")
    gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    for train_idx, test_idx in gss.split(df, groups=df["actor_id"].values):
        train_files = set(df.iloc[train_idx]["filename"])
        test_files = set(df.iloc[test_idx]["filename"])
        assert len(train_files.intersection(test_files)) == 0


def test_robustness_audio_perturbation_transforms():
    """Verify acoustic perturbation functions preserve signal shape and bounded ranges."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    # 1. Noise addition
    noisy_20 = add_noise(signal, snr_db=20.0)
    assert noisy_20.shape == signal.shape
    assert not np.isnan(noisy_20).any()
    assert not np.array_equal(noisy_20, signal)

    # 2. Volume Gain scaling
    gained = apply_gain(signal, gain_db=6.0)
    assert gained.shape == signal.shape
    assert np.max(np.abs(gained)) <= 1.0 # Protected from clipping

    attenuated = apply_gain(signal, gain_db=-12.0)
    assert np.max(np.abs(attenuated)) < np.max(np.abs(signal))

    # 3. Audio Truncation
    truncated_1s = truncate_audio(signal, sr=sr, max_sec=1.0)
    assert len(truncated_1s) == sr * 1.0
    assert len(truncated_1s) < len(signal)


def test_calibrated_temperature_scaling_and_abstention_gate(tmp_path):
    """Verify that low-confidence / high-entropy inputs trigger the selective abstention gate."""
    classes = ["angry", "calm", "happy", "sad"]
    rng = np.random.RandomState(42)

    # Train a minimal classifier
    X_train = rng.randn(60, 768).astype(np.float32)
    y_train = np.array([classes[i % len(classes)] for i in range(60)])

    clf = DeepEmotionClassifier(
        model_type="hubert",
        classifier_type="logistic_regression",
        confidence_threshold=0.35,
    )
    clf.fit(X_train, y_train)

    # Save with calibration metadata (temperature = 2.55)
    model_p = str(tmp_path / "calib_clf.joblib")
    meta_p = str(tmp_path / "calib_meta.json")
    clf.save(model_p, meta_p, extra_metadata={"temperature": 2.549})

    loaded_clf = DeepEmotionClassifier().load(model_p, meta_p)
    assert loaded_clf.metadata.get("temperature") == 2.549

    # Generate an ambiguous, near-uniform test vector (high uncertainty)
    ambiguous_vec = np.zeros(768, dtype=np.float32)
    res = loaded_clf.predict_single(ambiguous_vec)

    assert "emotion_label" in res
    assert "confidence" in res
    assert "is_reliable" in res
    assert "normalized_entropy" in res
    assert 0.0 <= res["normalized_entropy"] <= 1.0

    # Because probabilities are near uniform across 4 classes (~0.25 each < 0.35 threshold),
    # the model MUST abstain!
    assert res["is_reliable"] is False
    assert res["emotion_label"] == "uncertain"
    assert res["abstention_reason"] is not None
    assert "Unable to estimate vocal emotion reliably" in res["abstention_reason"] or "unclear vocal patterns" in res["abstention_reason"]


def test_explainability_acoustic_feature_dimensions():
    """Verify all 43 acoustic feature dimensions and names are consistent and non-empty."""
    assert len(FEATURE_NAMES) == 43
    assert len(set(FEATURE_NAMES)) == 43 # Strict uniqueness

    # Verify key acoustic feature groups are present
    assert any("pitch_f0" in f for f in FEATURE_NAMES)
    assert any("rms_energy" in f for f in FEATURE_NAMES)
    assert any("spectral_centroid" in f for f in FEATURE_NAMES)
    assert any("zero_crossing_rate" in f for f in FEATURE_NAMES)
    assert any("pause_fraction" in f for f in FEATURE_NAMES)


def test_transcription_normalization_and_devanagari_preservation():
    """Verify that normalization standardizes text without corrupting Devanagari or Latin Hinglish."""
    # Hindi text in Devanagari
    hindi_text = "आज मुझे बहुत अकेला महसूस हो रहा है।"
    norm_hi = normalize_text_for_eval(hindi_text)
    assert "आज" in norm_hi
    assert "अकेला" in norm_hi
    assert "महसूस" in norm_hi

    # Hinglish text
    hinglish_text = "Yaar, aaj mera mood bilkul theek nahi hai!"
    norm_hing = normalize_text_for_eval(hinglish_text)
    assert norm_hing == "yaar aaj mera mood bilkul theek nahi hai"

    # Perfect match test
    metrics = compute_metrics(norm_hing, norm_hing)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_language_detection_and_prompt_guardrails():
    """Verify language detection correctly tags Hinglish and prompt enforces language-matching and safety."""
    detector = LanguageDetector()

    # Hinglish input
    res_hing = detector.detect("Yaar I don't know kya karun, bohot stress ho raha hai.")
    assert res_hing.language_code == "hinglish"
    assert res_hing.is_code_switched is True

    # Hindi input
    res_hi = detector.detect("मुझे बहुत खुशी हो रही है आज।")
    assert res_hi.language_code == "hi"

    # System prompt check
    prompt_hing = build_system_prompt(target_language="hinglish", conversation_style="just_listen")
    assert "MANDATORY: The user is speaking Hinglish" in prompt_hing
    assert "Latin (Roman) script" in prompt_hing
    assert "Never translate what the user said back to them." in prompt_hing
    assert "NOT a therapist, medical doctor, or diagnostic tool" in prompt_hing
