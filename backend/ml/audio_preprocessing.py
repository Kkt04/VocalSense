"""
Audio Preprocessing for Speech Emotion Recognition.
Provides standardized audio loading, mono conversion, silence trimming,
sample rate standardization (16kHz), normalization, and spectrogram visualization.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

TARGET_SAMPLE_RATE = 16000


class AudioPreprocessingError(Exception):
    """Exception raised during audio preprocessing."""
    pass


def load_and_preprocess_audio(
    audio_source: Union[str, bytes, np.ndarray],
    target_sr: int = TARGET_SAMPLE_RATE,
    trim_silence: bool = True,
    top_db: float = 25.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load, resample, convert to mono, trim silence, and normalize audio.

    Args:
        audio_source: File path, bytes, or numpy array.
        target_sr: Standardized sample rate (default 16,000 Hz).
        trim_silence: Whether to trim leading/trailing silence.
        top_db: Silence threshold in decibels below peak.
        normalize: Whether to apply peak amplitude normalization.

    Returns:
        Tuple[np.ndarray, int]: (preprocessed_1d_float32_array, sample_rate)
    """
    try:
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise AudioPreprocessingError(f"Audio file does not exist: {audio_source}")
            y, sr = librosa.load(audio_source, sr=target_sr, mono=True)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source.astype(np.float32)
            sr = target_sr
            # Convert multi-channel to mono if necessary
            if y.ndim > 1:
                y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)
        elif isinstance(audio_source, bytes):
            import io
            import soundfile as sf
            data, sr_orig = sf.read(io.BytesIO(audio_source))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr_orig != target_sr:
                y = librosa.resample(data.astype(np.float32), orig_sr=sr_orig, target_sr=target_sr)
            else:
                y = data.astype(np.float32)
            sr = target_sr
        else:
            raise AudioPreprocessingError(f"Unsupported audio input type: {type(audio_source)}")

        if len(y) == 0:
            raise AudioPreprocessingError("Audio signal is empty (0 samples).")

        # Trim leading and trailing silence
        if trim_silence and len(y) > 512:
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
            # If trimming removed everything (e.g. nearly silent audio), keep original
            if len(y_trimmed) >= 512:
                y = y_trimmed

        # Peak normalization to [-1.0, 1.0] if signal has non-zero amplitude
        if normalize:
            max_val = np.max(np.abs(y))
            if max_val > 1e-5:
                y = y / max_val

        return y.astype(np.float32), sr

    except Exception as e:
        if isinstance(e, AudioPreprocessingError):
            raise
        raise AudioPreprocessingError(f"Audio preprocessing failed: {str(e)}")


def plot_waveform_and_spectrogram(
    audio_path: str,
    output_image_path: str = "backend/ml/eda_plots/waveform_spectrogram_sample.png",
    title: str = "Audio Waveform & Spectrogram Analysis",
) -> str:
    """
    Generate side-by-side plots of:
    1. Raw audio waveform
    2. Processed (trimmed & normalized) waveform
    3. Linear spectrogram (STFT)
    4. Mel spectrogram
    """
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # 1. Raw audio
    y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)

    # 2. Processed audio
    y_proc, sr_proc = load_and_preprocess_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)

    # 3. Compute STFT and Mel spectrogram
    d_proc = np.abs(librosa.stft(y_proc))
    db_linear = librosa.amplitude_to_db(d_proc, ref=np.max)

    mel_spec = librosa.feature.melspectrogram(y=y_proc, sr=sr_proc, n_mels=80)
    db_mel = librosa.power_to_db(mel_spec, ref=np.max)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Raw Waveform
    time_raw = np.linspace(0, len(y_raw) / sr_raw, num=len(y_raw))
    axes[0].plot(time_raw, y_raw, color="#8E7BAA", alpha=0.8)
    axes[0].set_title(f"1. Raw Audio Waveform (sr={sr_raw}Hz, dur={len(y_raw)/sr_raw:.2f}s)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Processed Waveform
    time_proc = np.linspace(0, len(y_proc) / sr_proc, num=len(y_proc))
    axes[1].plot(time_proc, y_proc, color="#70608C", alpha=0.9)
    axes[1].set_title(f"2. Processed Waveform (Trimmed & Normalized 16kHz, dur={len(y_proc)/sr_proc:.2f}s)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Linear STFT Spectrogram
    img1 = librosa.display.specshow(db_linear, sr=sr_proc, x_axis="time", y_axis="linear", ax=axes[2], cmap="magma")
    axes[2].set_title("3. Linear Frequency Spectrogram (STFT)")
    fig.colorbar(img1, ax=axes[2], format="%+2.0f dB")

    # Mel Spectrogram
    img2 = librosa.display.specshow(db_mel, sr=sr_proc, x_axis="time", y_axis="mel", ax=axes[3], cmap="viridis")
    axes[3].set_title("4. Mel Spectrogram (80 Mel Filterbanks)")
    fig.colorbar(img2, ax=axes[3], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()

    print(f"Visualization saved to {output_image_path}")
    return output_image_path
