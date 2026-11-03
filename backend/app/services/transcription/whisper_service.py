import time
import logging
from typing import Optional, Union, List
import numpy as np
from faster_whisper import WhisperModel
from backend.app.config import settings
from backend.app.schemas.transcription import (
    TranscriptionResponse,
    TranscriptionSegment,
)
from backend.app.services.transcription.audio_preprocessing import (
    AudioPreprocessor,
)
from backend.app.services.transcription.language_detection import (
    detect_language_characteristics,
)

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Manages the faster-whisper Automatic Speech Recognition (ASR) engine.
    Supports English, Hindi, and Hinglish with zero-shot code-switching preservation.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.configured_device = device or settings.WHISPER_DEVICE
        self.configured_compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        self.device = self._resolve_device(self.configured_device)
        self.compute_type = self._resolve_compute_type(self.configured_compute_type, self.device)
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _resolve_device(self, device_pref: str) -> str:
        """Resolve hardware execution device (cuda or cpu)."""
        if device_pref == "cuda":
            return "cuda"
        elif device_pref == "cpu":
            return "cpu"
        # Auto-detect CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _resolve_compute_type(self, compute_pref: str, resolved_device: str) -> str:
        """Resolve numeric precision for CTranslate2 inference."""
        if compute_pref != "auto":
            return compute_pref
        if resolved_device == "cuda":
            return "float16"
        # On CPU, int8 provides optimal latency and low memory footprint
        return "int8"

    def _load_model(self):
        """Instantiate and load the CTranslate2 Whisper model."""
        try:
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})..."
            )
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}", exc_info=True)
            self.model = None
            raise

    def is_ready(self) -> bool:
        """Check whether the model is loaded and ready for inference."""
        return self.model is not None

    def transcribe(
        self,
        audio_input: Union[bytes, str, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio input into text preserving spoken language and code-switching.

        Args:
            audio_input: Audio bytes, file path, or 16kHz float32 numpy array.
            language: Optional language code ('en', 'hi', or None for auto-detect).
            initial_prompt: Conditioning prompt for vocabulary and spelling.

        Returns:
            TranscriptionResponse: Structured transcript with timestamps and language.
        """
        if not self.is_ready():
            self._load_model()

        start_time = time.perf_counter()

        # Step 1: Preprocess and decode audio into 16kHz mono float32 array
        if isinstance(audio_input, np.ndarray):
            audio_array = audio_input
            duration = float(len(audio_array)) / 16000.0
        else:
            audio_array, duration = AudioPreprocessor.load_and_resample(audio_input)

        # Step 2: Configure Whisper transcription options
        # Task is strictly 'transcribe' (NOT 'translate') to preserve the spoken language!
        lang_arg = None if (language in (None, "", "auto")) else language
        if initial_prompt is not None:
            prompt = initial_prompt
        elif lang_arg == "hi":
            prompt = "नमस्ते, यह हिंदी में बातचीत है।"
        else:
            prompt = settings.WHISPER_INITIAL_PROMPT

        # Step 3: Run inference via faster-whisper
        segments_gen, info = self.model.transcribe(
            audio_array,
            task="transcribe",
            language=lang_arg,
            beam_size=settings.WHISPER_BEAM_SIZE,
            initial_prompt=prompt,
            vad_filter=True,  # Voice activity detection to trim non-speech noise
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_lang = info.language
        lang_prob = round(float(info.language_probability), 4)

        # Collect segment data
        segments: List[TranscriptionSegment] = []
        transcript_parts = []

        for seg in segments_gen:
            clean_seg_text = seg.text.strip()
            if clean_seg_text:
                transcript_parts.append(clean_seg_text)
                segments.append(
                    TranscriptionSegment(
                        start=round(float(seg.start), 2),
                        end=round(float(seg.end), 2),
                        text=clean_seg_text,
                        avg_logprob=round(float(seg.avg_logprob), 3) if seg.avg_logprob is not None else None,
                        no_speech_prob=round(float(seg.no_speech_prob), 3) if seg.no_speech_prob is not None else None,
                    )
                )

        consolidated_transcript = " ".join(transcript_parts).strip()

        # Step 4: Analyze multilingual & code-switching characteristics (Hinglish detection)
        all_probs = info.all_language_probs if hasattr(info, "all_language_probs") else None
        final_lang, is_hinglish, _ = detect_language_characteristics(
            whisper_detected_lang=detected_lang,
            language_prob=lang_prob,
            transcript_text=consolidated_transcript,
            all_language_probs=dict(all_probs) if all_probs else None,
        )

        processing_time = round(time.perf_counter() - start_time, 3)

        return TranscriptionResponse(
            success=True,
            transcript=consolidated_transcript,
            detected_language=final_lang,
            language_probability=lang_prob,
            duration=round(duration, 2),
            segments=segments,
            is_hinglish=is_hinglish,
            model_used=f"faster-whisper-{self.model_size}",
            processing_time_sec=processing_time,
            message="Audio transcribed successfully.",
        )


# Global singleton instance
_whisper_instance: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """Retrieve or initialize the global WhisperService singleton."""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperService()
    return _whisper_instance
