"""
Acoustic Feature Extraction for Speech Emotion Recognition (SER).
Extracts reproducible acoustic features:
- MFCCs (13 coefficients: mean and std)
- Pitch / F0 (mean, std, min, max, voiced fraction via pyin/yin)
- RMS Energy (mean, std, max)
- Spectral Centroid (mean, std)
- Spectral Bandwidth (mean, std)
- Spectral Rolloff (mean, std)
- Zero-Crossing Rate (mean, std)
- Temporal / Pause statistics (pause fraction, vocal duration)
"""

import os
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import librosa
import pandas as pd
from backend.ml.audio_preprocessing import load_and_preprocess_audio, TARGET_SAMPLE_RATE

# Feature names in deterministic order
FEATURE_NAMES: List[str] = (
    [f"mfcc_{i+1}_mean" for i in range(13)]
    + [f"mfcc_{i+1}_std" for i in range(13)]
    + [
        "pitch_f0_mean",
        "pitch_f0_std",
        "pitch_f0_min",
        "pitch_f0_max",
        "voiced_fraction",
        "rms_energy_mean",
        "rms_energy_std",
        "rms_energy_max",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "pause_fraction",
    ]
)


def extract_pitch_f0(
    y: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 400.0,
) -> Tuple[float, float, float, float, float]:
    """
    Extract fundamental frequency (F0) using YIN / pyin algorithm.
    Returns:
        (f0_mean, f0_std, f0_min, f0_max, voiced_fraction)
    """
    try:
        # librosa.yin is fast and robust for speech F0 extraction
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=256)

        # Filter out unvoiced / out-of-range frames
        voiced = f0[(f0 >= fmin) & (f0 <= fmax) & (~np.isnan(f0))]

        if len(voiced) > 0:
            f0_mean = float(np.mean(voiced))
            f0_std = float(np.std(voiced))
            f0_min = float(np.min(voiced))
            f0_max = float(np.max(voiced))
            voiced_fraction = float(len(voiced) / len(f0))
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_min = 0.0
            f0_max = 0.0
            voiced_fraction = 0.0

        return f0_mean, f0_std, f0_min, f0_max, voiced_fraction
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def extract_acoustic_features(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
) -> Dict[str, float]:
    """
    Extract acoustic feature dictionary from preprocessed audio signal.
    """
    if len(y) < 512:
        # Audio too short: return zeros for all features
        return {name: 0.0 for name in FEATURE_NAMES}

    features: Dict[str, float] = {}

    # 1. MFCCs (13 coefficients, mean & std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # 2. Pitch / F0
    f0_mean, f0_std, f0_min, f0_max, voiced_fraction = extract_pitch_f0(y, sr)
    features["pitch_f0_mean"] = f0_mean
    features["pitch_f0_std"] = f0_std
    features["pitch_f0_min"] = f0_min
    features["pitch_f0_max"] = f0_max
    features["voiced_fraction"] = voiced_fraction

    # 3. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    features["rms_energy_mean"] = float(np.mean(rms))
    features["rms_energy_std"] = float(np.std(rms))
    features["rms_energy_max"] = float(np.max(rms)) if len(rms) > 0 else 0.0

    # 4. Spectral Features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
    features["spectral_centroid_mean"] = float(np.mean(cent))
    features["spectral_centroid_std"] = float(np.std(cent))

    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(band))
    features["spectral_bandwidth_std"] = float(np.std(band))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=1024, hop_length=256)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=1024, hop_length=256)[0]
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"] = float(np.std(zcr))

    # 5. Temporal / Pause statistics
    # Frames where energy is less than 15% of average RMS are treated as pauses/silence
    mean_energy = features["rms_energy_mean"]
    pause_frames = np.sum(rms < (mean_energy * 0.15)) if mean_energy > 0 else len(rms)
    pause_fraction = float(pause_frames / len(rms)) if len(rms) > 0 else 1.0
    features["pause_fraction"] = pause_fraction

    # Clean NaNs and Infs
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features


def extract_features_vector(
    audio_source: Union[str, bytes, np.ndarray],
) -> np.ndarray:
    """
    Extract a deterministic 1D numpy array of features in the exact order of FEATURE_NAMES.
    """
    if isinstance(audio_source, (str, bytes)):
        y, sr = load_and_preprocess_audio(audio_source)
    else:
        y = audio_source
        sr = TARGET_SAMPLE_RATE

    feat_dict = extract_acoustic_features(y, sr)
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_dataset_features(
    manifest_csv: str = "data/ravdess/metadata.csv",
    output_features_csv: str = "data/ravdess/features_ravdess.csv",
) -> pd.DataFrame:
    """
    Extract acoustic features across the entire RAVDESS dataset manifest
    and export to a structured CSV.
    """
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found at {manifest_csv}")

    df_meta = pd.read_csv(manifest_csv)
    records: List[Dict[str, Any]] = []

    print(f"Extracting acoustic features for {len(df_meta)} recordings...")
    for idx, row in df_meta.iterrows():
        file_path = row["file_path"]
        try:
            y, sr = load_and_preprocess_audio(file_path)
            feats = extract_acoustic_features(y, sr)
            # Combine metadata identifiers with acoustic features
            record = {
                "file_path": file_path,
                "filename": row["filename"],
                "actor_id": row["actor_id"],
                "gender": row["gender"],
                "emotion_id": row["emotion_id"],
                "emotion_label": row["emotion_label"],
                "intensity": row["intensity"],
                "duration_sec": row["duration_sec"],
                **feats,
            }
            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to extract features for {file_path}: {e}")

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_meta):
            print(f"Processed {idx + 1}/{len(df_meta)} files...")

    df_feats = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_features_csv), exist_ok=True)
    df_feats.to_csv(output_features_csv, index=False)
    print(f"Features extracted and saved to {output_features_csv} ({len(df_feats)} rows, {len(FEATURE_NAMES)} features).")

    return df_feats
