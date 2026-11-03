import os
from typing import Tuple, Optional
from fastapi import UploadFile
from backend.app.config import settings


class AudioValidationError(Exception):
    """Exception raised for errors in audio validation."""

    def __init__(self, message: str, code: str = "AUDIO_VALIDATION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AudioValidator:
    """Validates uploaded audio files for MIME type, extension, size, and header magic numbers."""

    # File magic numbers / headers
    MAGIC_SIGNATURES = {
        "wav": [b"RIFF"],  # Note: byte 8-12 should be b"WAVE"
        "webm": [b"\x1a\x45\xdf\xa3"],  # Matroska / EBML header
        "ogg": [b"OggS"],
        "mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
        "mp4_m4a": [b"ftyp"],  # typically at offset 4
    }

    @classmethod
    def detect_format_from_bytes(cls, header: bytes) -> Optional[str]:
        """Detect audio container format from initial file bytes."""
        if len(header) < 4:
            return None

        # Check WAV (RIFF....WAVE)
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
            return "wav"

        # Check WebM / Matroska
        if header.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"

        # Check OGG
        if header.startswith(b"OggS"):
            return "ogg"

        # Check MP3 ID3 header or sync frames
        if header.startswith(b"ID3") or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return "mp3"

        # Check MP4 / M4A (ftyp at offset 4)
        if len(header) >= 8 and header[4:8] == b"ftyp":
            return "m4a"

        return None

    @classmethod
    async def validate_upload(
        cls, file: UploadFile
    ) -> Tuple[bytes, str, int]:
        """
        Validate an uploaded audio file.

        Returns:
            Tuple[bytes, str, int]: (file_bytes, detected_format, size_bytes)

        Raises:
            AudioValidationError if the file fails validation.
        """
        if not file:
            raise AudioValidationError("No audio file was uploaded.", code="MISSING_FILE")

        filename = file.filename or "recording.webm"
        content_type = (file.content_type or "").lower().split(";")[0].strip()

        # 1. Validate file extension (if extension exists)
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_AUDIO_EXTENSIONS)
            raise AudioValidationError(
                f"File extension '{ext}' is not supported. Allowed formats: {allowed}",
                code="UNSUPPORTED_EXTENSION",
            )

        # 2. Read contents up to max size + 1 to detect oversize efficiently
        max_bytes = settings.MAX_AUDIO_SIZE_BYTES
        # Read the file stream in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        total_read = 0
        chunks = []

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_bytes:
                max_mb = max_bytes / (1024 * 1024)
                raise AudioValidationError(
                    f"Audio file exceeds maximum allowed size of {max_mb:.1f} MB.",
                    code="FILE_TOO_LARGE",
                )
            chunks.append(chunk)

        if total_read == 0:
            raise AudioValidationError(
                "Uploaded audio file is empty (0 bytes).", code="EMPTY_FILE"
            )

        content = b"".join(chunks)

        # 3. Detect format from magic bytes
        header = content[:16]
        detected_format = cls.detect_format_from_bytes(header)

        # Fallback to extension or content-type if magic bytes are ambiguous
        # (e.g. raw audio stream chunks or browser-specific blobs)
        if not detected_format:
            if ext:
                detected_format = ext.lstrip(".")
            elif "webm" in content_type:
                detected_format = "webm"
            elif "wav" in content_type:
                detected_format = "wav"
            elif "ogg" in content_type:
                detected_format = "ogg"
            elif "mp4" in content_type or "m4a" in content_type:
                detected_format = "m4a"
            elif "mpeg" in content_type or "mp3" in content_type:
                detected_format = "mp3"
            else:
                detected_format = "unknown"

        # 4. Validate MIME type if provided and not generic
        if content_type and content_type not in ("application/octet-stream", ""):
            if content_type not in settings.ALLOWED_AUDIO_MIME_TYPES:
                # Also accept generic audio/*
                if not content_type.startswith("audio/"):
                    raise AudioValidationError(
                        f"MIME type '{content_type}' is not recognized as a valid audio format.",
                        code="INVALID_MIME_TYPE",
                    )

        return content, detected_format, total_read
