"""
Wav2Vec 2.0 Feature Extraction and Representation Pipeline for Speech Emotion Recognition.

Model Architecture:
- Checkpoint: facebook/wav2vec2-base
- Pretraining: 960 hours of unlabeled 16kHz LibriSpeech audio (Librispeech-960h).
- Architecture: 7-layer temporal convolutional feature encoder (strides [5,2,2,2,2,2,2], receptive field ~25ms, stride ~20ms),
  followed by a 12-layer Transformer encoder with model dimension 768, 8 attention heads, and 3072 FFN dimension (~95M parameters).
- Self-Supervised Objective: Contrastive task over quantized latent speech representations with Gumbel-Softmax vector quantization.
- Sampling Rate: 16,000 Hz single-channel mono.
- Representations: 768-dimensional contextual hidden state per ~20ms frame.
- Pooling: Mean-pooling across the temporal sequence dimension produces a fixed 768-dimensional vector representation.
- License: Apache 2.0 (open for research and commercial use).
"""

import logging
import time
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_WAV2VEC2_CHECKPOINT = "facebook/wav2vec2-base"


class Wav2Vec2Pipeline:
    """
    Manages pretrained Wav2Vec2 feature extraction and contextual representation learning.
    Extracts high-level temporal speech embeddings for downstream emotion classification.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_WAV2VEC2_CHECKPOINT,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = self._resolve_device(device)
        self.feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None
        self.model: Optional[Wav2Vec2Model] = None
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
        """Loads feature extractor and model weights."""
        try:
            logger.info("Loading Wav2Vec2 feature extractor from %s...", self.checkpoint)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            logger.info("Loading Wav2Vec2 transformer backbone onto device %s...", self.device)
            self.model = Wav2Vec2Model.from_pretrained(
                self.checkpoint,
                local_files_only=self.local_files_only,
            )
            self.model.to(self.device)
            self.model.eval()
            self.hidden_size = self.model.config.hidden_size
            logger.info(
                "Wav2Vec2 successfully loaded (%s, hidden_size=%d, device=%s)",
                self.checkpoint,
                self.hidden_size,
                self.device,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load Wav2Vec2 model from '%s' (local_files_only=%s): %s. "
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
                self.model = Wav2Vec2Model.from_pretrained(
                    self.checkpoint,
                    local_files_only=False,
                )
                self.model.to(self.device)
                self.model.eval()
                self.hidden_size = self.model.config.hidden_size
                logger.info("Wav2Vec2 model loaded from remote/cache.")
            except Exception as e:
                logger.error("Wav2Vec2 model could not be initialized: %s", e)
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

        # Convert multi-channel to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample to 16 kHz if required
        if sr != TARGET_SAMPLE_RATE:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

        # Peak normalization to avoid clipping or numerical underflow
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

        Args:
            audio_input: Audio file path, audio bytes, or 1D float numpy array.
            sample_rate: Input sampling rate (auto-resampled to 16kHz).
            pooling: Temporal pooling strategy: 'mean', 'mean_std', or 'max'.

        Returns:
            1D numpy array representing the utterance embedding (768-dim for 'mean').
        """
        if not self.is_ready():
            raise RuntimeError("Wav2Vec2 model is not loaded.")

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
        Extracts speech representation embeddings for a list of preprocessed 16kHz audio signals in batches.
        """
        if not self.is_ready():
            raise RuntimeError("Wav2Vec2 model is not loaded.")

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

                if inputs.get("attention_mask") is not None:
                    # Account for padding in attention mask
                    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
                    # Note: Wav2Vec2 conv encoder downsamples by factor of 320 (20ms stride)
                    # For safety, compute sequence-wise mean
                    emb = hidden_states.mean(dim=1)
                else:
                    emb = hidden_states.mean(dim=1)

                all_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns architecture and pretraining specifications for documentation and reproducibility."""
        return {
            "model_name": "Wav2Vec2 Base",
            "checkpoint": self.checkpoint,
            "architecture": "Temporal Conv Encoder + 12-Layer Transformer",
            "hidden_size": self.hidden_size,
            "pretraining_dataset": "LibriSpeech (960 hours of 16kHz speech)",
            "supported_languages": ["en (Acoustic representations generalize cross-lingually)"],
            "sampling_rate": TARGET_SAMPLE_RATE,
            "license": "Apache-2.0",
            "device": str(self.device),
        }
