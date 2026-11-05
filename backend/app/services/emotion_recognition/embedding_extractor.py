"""
Unified Embedding Extractor and Caching Engine.
Extracts deep speech embeddings for datasets using Wav2Vec2 and HuBERT,
caching results to disk to avoid redundant forward passes during experimental sweeps.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.app.services.emotion_recognition.wav2vec2_model import Wav2Vec2Pipeline
from backend.app.services.emotion_recognition.hubert_model import HuBERTPipeline

logger = logging.getLogger(__name__)

CACHE_DIR = "data/embeddings"


class DatasetEmbeddingExtractor:
    """
    Coordinates dataset-level embedding extraction and disk caching.
    Supports Wav2Vec2 and HuBERT.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, model_type: str, dataset_name: str = "ravdess") -> str:
        return os.path.join(self.cache_dir, f"{model_type}_{dataset_name}.npz")

    def load_cached_embeddings(
        self,
        model_type: str,
        dataset_name: str = "ravdess",
    ) -> Optional[Dict[str, np.ndarray]]:
        """Loads precomputed embeddings from disk cache if available."""
        path = self.get_cache_path(model_type, dataset_name)
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                logger.info("Loaded cached embeddings from %s", path)
                return {
                    "embeddings": data["embeddings"],
                    "labels": data["labels"],
                    "actor_ids": data["actor_ids"],
                    "filenames": data["filenames"],
                }
            except Exception as e:
                logger.warning("Failed to load cached embeddings from %s: %s", path, e)
        return None

    def save_cached_embeddings(
        self,
        model_type: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        actor_ids: np.ndarray,
        filenames: np.ndarray,
        dataset_name: str = "ravdess",
    ) -> str:
        """Saves extracted embeddings and metadata to disk."""
        path = self.get_cache_path(model_type, dataset_name)
        np.savez_compressed(
            path,
            embeddings=embeddings,
            labels=labels,
            actor_ids=actor_ids,
            filenames=filenames,
        )
        logger.info("Saved %d embeddings to %s (size: %.2f MB)", len(embeddings), path, os.path.getsize(path) / (1024 * 1024))
        return path

    def extract_or_load(
        self,
        model_type: str,  # "wav2vec2" or "hubert"
        manifest_df: pd.DataFrame,
        pipeline: Optional[Any] = None,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Retrieves cached embeddings or extracts them across all rows of the dataset manifest.
        """
        if not force_recompute:
            cached = self.load_cached_embeddings(model_type)
            if cached is not None and len(cached["embeddings"]) == len(manifest_df):
                return cached

        logger.info("Extracting %s embeddings for %d recordings...", model_type, len(manifest_df))
        if pipeline is None:
            if model_type == "wav2vec2":
                pipeline = Wav2Vec2Pipeline()
            elif model_type == "hubert":
                pipeline = HuBERTPipeline()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        embeddings_list: List[np.ndarray] = []
        labels_list: List[str] = []
        actors_list: List[int] = []
        filenames_list: List[str] = []

        start_time = time.time()
        for idx, row in manifest_df.iterrows():
            file_path = row["file_path"]
            emb = pipeline.extract_embedding(file_path)
            embeddings_list.append(emb)
            labels_list.append(str(row["emotion_label"]))
            actors_list.append(int(row["actor_id"]))
            filenames_list.append(str(row["filename"]))

            if (idx + 1) % 100 == 0 or (idx + 1) == len(manifest_df):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                logger.info(
                    "[%s] Processed %d/%d (%.1f%%) — %.2f samples/sec",
                    model_type,
                    idx + 1,
                    len(manifest_df),
                    100.0 * (idx + 1) / len(manifest_df),
                    rate,
                )

        X = np.vstack(embeddings_list)
        y = np.array(labels_list)
        groups = np.array(actors_list)
        filenames = np.array(filenames_list)

        self.save_cached_embeddings(
            model_type=model_type,
            embeddings=X,
            labels=y,
            actor_ids=groups,
            filenames=filenames,
        )

        return {
            "embeddings": X,
            "labels": y,
            "actor_ids": groups,
            "filenames": filenames,
        }
