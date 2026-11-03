import io
import av
import numpy as np
from typing import Tuple, Union


class PreprocessingError(Exception):
    """Exception raised during audio preprocessing and decoding."""

    def __init__(self, message: str, code: str = "PREPROCESSING_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AudioPreprocessor:
    """
    Decodes diverse audio container formats (WAV, WebM, MP3, M4A, OGG)
    and resamples to 16,000 Hz single-channel (mono) float32 array
    as required by Whisper's log-Mel spectrogram feature extractor.
    """

    TARGET_SAMPLE_RATE = 16000

    @classmethod
    def load_and_resample(
        cls,
        audio_source: Union[bytes, str],
        min_duration_sec: float = 0.2,
        max_duration_sec: float = 600.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Decode and resample audio into a normalized 16kHz mono float32 array.

        Args:
            audio_source: In-memory audio bytes or string path to audio file.
            min_duration_sec: Minimum audio duration allowed.
            max_duration_sec: Maximum audio duration allowed.

        Returns:
            Tuple[np.ndarray, float]: (audio_array_16khz, duration_in_seconds)

        Raises:
            PreprocessingError: If audio is corrupted, empty, or outside limits.
        """
        try:
            if isinstance(audio_source, bytes):
                if len(audio_source) == 0:
                    raise PreprocessingError("Audio buffer is empty (0 bytes).", code="EMPTY_AUDIO")
                container = av.open(io.BytesIO(audio_source))
            else:
                container = av.open(audio_source)
        except Exception as e:
            raise PreprocessingError(f"Failed to open audio container: {str(e)}", code="CONTAINER_DECODE_ERROR")

        try:
            # Check for audio streams
            if not container.streams.audio:
                raise PreprocessingError("No audio stream found in the uploaded file.", code="NO_AUDIO_STREAM")

            audio_stream = container.streams.audio[0]

            # Setup PyAV AudioResampler for 16kHz mono float32
            resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=cls.TARGET_SAMPLE_RATE,
            )

            frames = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames:
                        for rf in resampled_frames:
                            # Convert plane to numpy array
                            array = rf.to_ndarray()
                            frames.append(array.flatten())

            # Flush any remaining resampler frames
            resampled_frames = resampler.resample(None)
            if resampled_frames:
                for rf in resampled_frames:
                    array = rf.to_ndarray()
                    frames.append(array.flatten())

        except PreprocessingError:
            raise
        except Exception as e:
            raise PreprocessingError(f"Error decoding audio frames: {str(e)}", code="FRAME_DECODE_ERROR")
        finally:
            container.close()

        if not frames:
            raise PreprocessingError("Decoded audio contains no valid audio frames.", code="NO_FRAMES_DECODED")

        # Concatenate all frame chunks into a single 1D numpy array
        audio_array = np.concatenate(frames).astype(np.float32)

        # Duration in seconds
        duration = float(len(audio_array)) / float(cls.TARGET_SAMPLE_RATE)

        if duration < min_duration_sec:
            raise PreprocessingError(
                f"Audio recording is too short ({duration:.2f}s). Minimum required is {min_duration_sec:.1f}s.",
                code="AUDIO_TOO_SHORT",
            )

        if duration > max_duration_sec:
            raise PreprocessingError(
                f"Audio recording exceeds maximum allowed duration ({duration:.1f}s > {max_duration_sec:.1f}s).",
                code="AUDIO_TOO_LONG",
            )

        # Basic energy check: warn or reject if pure silence (RMS < 1e-4)
        rms = np.sqrt(np.mean(np.square(audio_array)))
        if rms < 1e-4:
            # Still return audio array, but flag could be checked
            pass

        return audio_array, duration
