import os
import uuid
import logging
from typing import Optional
from backend.app.config import settings

logger = logging.getLogger(__name__)


class AudioStorageService:
    """
    Manages ephemeral and privacy-first storage for user audio files.

    In accordance with VocalSense privacy principles:
    - Raw audio is not permanently stored by default.
    - Temporary processing files can be created and cleaned up.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.TEMP_STORAGE_DIR
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure the temp storage directory exists."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create storage directory {self.base_dir}: {e}")

    def save_ephemeral(self, content: bytes, ext: str = "webm") -> str:
        """
        Save audio bytes to a temporary file.

        Returns:
            str: Path to the temporary file.
        """
        self._ensure_dir()
        filename = f"{uuid.uuid4().hex}.{ext.lstrip('.')}"
        file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    def delete_file(self, file_path: str) -> bool:
        """Safely delete an audio file."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            logger.warning(f"Error removing temporary audio file {file_path}: {e}")
        return False

    def handle_upload_privacy(self, file_path: Optional[str]) -> None:
        """
        Enforce privacy policy: delete the temporary audio immediately
        if RETAIN_RAW_AUDIO is disabled.
        """
        if not settings.RETAIN_RAW_AUDIO and file_path:
            self.delete_file(file_path)
