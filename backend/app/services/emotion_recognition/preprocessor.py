"""
Preprocessing module for runtime emotion prediction.
Ensures runtime audio undergoes identical transformations to training data.
"""

from typing import Tuple, Union
import numpy as np
from backend.ml.audio_preprocessing import (
    load_and_preprocess_audio,
    AudioPreprocessingError,
    TARGET_SAMPLE_RATE,
)

__all__ = ["load_and_preprocess_audio", "AudioPreprocessingError", "TARGET_SAMPLE_RATE"]
