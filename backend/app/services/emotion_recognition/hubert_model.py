"""
HuBERT (Hidden-Unit BERT) Feature Extraction and Representation Pipeline for Speech Emotion Recognition.

Model Architecture:
- Checkpoint: facebook/hubert-base-ls960
- Pretraining: 960 hours of unlabeled 16kHz LibriSpeech audio (Librispeech-960h).
- Architecture: 7-layer temporal convolutional feature encoder, followed by a 12-layer Transformer
  encoder with model dimension 768, 8 attention heads, and 3072 FFN dimension (~95M parameters).
- Self-Supervised Objective:
  Unlike Wav2Vec2 (which utilizes contrastive loss over quantized representations), HuBERT utilizes
  masked prediction of discrete acoustic cluster units (generated via offline k-means clustering).
  This forces the Transformer to learn both acoustic and pseudo-phonetic structures over masked spans.
- Sampling Rate: 16,000 Hz single-channel mono.
- Representations: 768-dimensional contextual hidden state per frame.
- Pooling: Mean-pooling across the temporal sequence produces a fixed 768-dimensional embedding.
- License: MIT License (Meta AI).
"""

import logging
import time
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_HUBERT_CHECKPOINT = "facebook/hubert-base-ls960"


class HuBERTPipeline:
    """
    Manages pretrained HuBERT feature extraction and contextual representation learning.
    Extracts high-level masked-prediction representations for downstream emotion classification.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_HUBERT_CHECKPOINT,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = self._resolve_device(device)
        self.feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None
        self.model: Optional[HubertModel] = None
        self.hidden_size: int = 768
        self.local_files_only = local_files_only
        self._load_model()

    def _resolve_device(self, preferred_device: Optional[str]) -> torch.device:
        if preferred_device and preferred_device != "auto":
            return torch.device(preferred_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> None:
        """Loads HuBERT feature extractor and model weights."""
        try:
            logger.info("Loading HuBERT feature extractor from %s...", self.checkpoint)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            logger.info("Loading HuBERT transformer backbone onto device %s...", self.device)
            self.model = HubertModel.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            self.model.to(self.device)
            self.model.eval()
            self.hidden_size = self.model.config.hidden_size
            logger.info(
                "HuBERT successfully loaded (%s, hidden_size=%d, device=%s)",
                self.checkpoint,
                self.hidden_size,
                self.device,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load HuBERT model from '%s' (local_files_only=%s): %s. "
                "Retrying with local_files_only=False if network allows...",
                self.checkpoint,
                self.local_files_only,
                exc,
            )
            try:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.checkpoint,
                    local_files_only=False,
                )
                self.model = HubertModel.from_pretrained(
                    self.checkpoint,
                    local_files_only=False,
                )
                self.model.to(self.device)
                self.model.eval()
                self.hidden_size = self.model.config.hidden_size
                logger.info("HuBERT model loaded from remote/cache.")
            except Exception as e:
                logger.error("HuBERT model could not be initialized: %s", e)
                self.model = None
                self.feature_extractor = None

    def is_ready(self) -> bool:
        return self.model is not None and self.feature_extractor is not None

    def preprocess_audio(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Normalizes and resamples arbitrary audio inputs to 16,000 Hz single-channel mono.
        """
        if isinstance(audio_input, str):
            y, sr = sf.read(audio_input)
        elif isinstance(audio_input, bytes):
            import io
            y, sr = sf.read(io.BytesIO(audio_input))
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
            sr = sample_rate or TARGET_SAMPLE_RATE
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr != TARGET_SAMPLE_RATE:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / max_abs

        return y.astype(np.float32)

    def extract_embedding(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extracts contextual speech representation embedding for a single audio recording.
        """
        if not self.is_ready():
            raise RuntimeError("HuBERT model is not loaded.")

        y = self.preprocess_audio(audio_input, sample_rate)

        inputs = self.feature_extractor(
            y,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

            if pooling == "mean":
                embedding = hidden_states.mean(dim=1).squeeze(0)  # [768]
            elif pooling == "mean_std":
                mean = hidden_states.mean(dim=1).squeeze(0)
                std = hidden_states.std(dim=1).squeeze(0)
                embedding = torch.cat([mean, std], dim=0)  # [1536]
            elif pooling == "max":
                embedding = hidden_states.max(dim=1).values.squeeze(0)
            else:
                embedding = hidden_states.mean(dim=1).squeeze(0)

        return embedding.detach().cpu().numpy().astype(np.float32)

    def extract_batch_embeddings(
        self,
        audio_list: List[np.ndarray],
        batch_size: int = 16,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extracts HuBERT speech representation embeddings for a list of audio signals in batches.
        """
        if not self.is_ready():
            raise RuntimeError("HuBERT model is not loaded.")

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(audio_list), batch_size):
            batch_audio = audio_list[i : i + batch_size]
            inputs = self.feature_extractor(
                batch_audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # [B, seq_len, 768]
                emb = hidden_states.mean(dim=1)
                all_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns architecture and pretraining specifications for HuBERT."""
        return {
            "model_name": "HuBERT Base",
            "checkpoint": self.checkpoint,
            "architecture": "Temporal Conv Encoder + 12-Layer Transformer with Masked Unit Prediction",
            "hidden_size": self.hidden_size,
            "pretraining_dataset": "LibriSpeech (960 hours of 16kHz speech)",
            "supported_languages": ["en (Acoustic representations generalize cross-lingually)"],
            "sampling_rate": TARGET_SAMPLE_RATE,
            "license": "MIT",
            "device": str(self.device),
        }
